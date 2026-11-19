"""Evaluate Mol-Instructions SFT-native predictions and OpenCompass-strict scores.

Both scoring paths use the same underlying metrics:
 * FS/MG/RP/RS: Morgan (radius 2, 2048 bits) Tanimoto * 100, invalid = 0.
 * MC: nltk METEOR on .split() tokens, 0-1.
 * PP: MAE; missing/unparsable answers fall back to prediction 0.

Primary 'score' uses native SFT output parsing (untagged SELFIES/plain numbers).
Also saves 'opencompass_strict_score' without format adaptation for diagnostics.
Neither is a verified reproduction of the unpublished InternLM aggregate.
"""
import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime

from mol_instructions_common import TASK_ORDER, TASKS, read_test

CHEM = {'FS', 'MG', 'RP', 'RS'}
TAG_PATTERN = re.compile(r'<(SELFIES|SMILES)>(.*?)</\1>', re.DOTALL)
NUMBER_PATTERN = re.compile(r'(?:<NUMBER>\s*|\\boxed\{)\s*(-?\d*\.?\d+)\s*(?:</NUMBER>|\})')
PLAIN_NUMBER = re.compile(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?')
THINK_PATTERN = re.compile(r'<think>.*?</think>', re.DOTALL)


def remove_reasoning(text):
    """Mirror OpenCompass extract_non_reasoning_content."""
    if '<think>' not in text and '</think>' in text:
        return text.split('</think>')[-1].strip()
    return THINK_PATTERN.sub('', text).strip()


def extract_chem_tag(text):
    matches = TAG_PATTERN.findall(text)
    return matches[-1] if matches else (None, None)


def extract_number(text):
    values = NUMBER_PATTERN.findall(text)
    return float(values[-1]) if values else None


def score_prediction(x, generator):
    task = x['task']
    pred = remove_reasoning(str(x.get('prediction', '')))
    ans = str(x['reference'])
    if task in CHEM:
        import selfies as sf
        from rdkit import Chem, DataStructs
        pred_type, pred_txt = extract_chem_tag(pred)
        # Official fallback: if no tagged representation, assume SMILES.
        if not pred_txt:
            pred_type, pred_txt = 'SMILES', pred
        # Ground truth in uploaded dataset is raw SELFIES, so wrap it in
        # official expected <SELFIES> tags before applying same logic.
        ans_type, ans_txt = 'SELFIES', ans
        # Official evaluator decodes both together if (and only if) prediction
        # is tagged as SELFIES; otherwise it falls back to raw SMILES for both.
        if pred_type == 'SELFIES':
            try:
                pred_smiles, ans_smiles = sf.decoder(pred_txt), sf.decoder(ans_txt)
            except Exception:
                return {'score': 0.0, 'valid': False,
                        'parsed_prediction': pred_txt, 'parsed_reference': ans_txt,
                        'reason': 'selfies_decode_failed'}
        else:
            # This emulates the official FTSEvaluator behavior for untagged
            # predictions: absent <SELFIES> tags, raw reference is not decoded.
            # The reference here is raw SELFIES, not SMILES.
            pred_smiles, ans_smiles = pred_txt, ans_txt
        mol1 = Chem.MolFromSmiles(pred_smiles)
        mol2 = Chem.MolFromSmiles(ans_smiles)
        if mol1 is None or mol2 is None:
            return {'score': 0.0, 'valid': False,
                    'parsed_prediction': pred_smiles, 'parsed_reference': ans_smiles,
                    'reason': 'invalid_smiles'}
        fp1, fp2 = generator.GetFingerprint(mol1), generator.GetFingerprint(mol2)
        score = float(DataStructs.TanimotoSimilarity(fp1, fp2) * 100)
        return {'score': score, 'valid': True,
                'parsed_prediction': pred_smiles, 'parsed_reference': ans_smiles}
    if task == 'MC':
        from nltk.translate.meteor_score import meteor_score
        score = float(meteor_score([ans.split()], pred.split())) if ans and pred else 0.0
        return {'score': score, 'parsed_prediction': pred}
    if task == 'PP':
        num = extract_number(pred)
        gold = float(ans)
        # The official MAEEvaluator uses "if not pred: pred = 0.0".
        value = 0.0 if not num else num
        return {'score': abs(value - gold), 'parsed_prediction': value,
                'number_extracted': num is not None}
    raise ValueError(f'Unexpected task {task}')


def score_native_prediction(x, generator):
    """Extract a final answer conservatively, then apply original task metrics.

    - An unambiguous structure is required for chemical tasks: SELFIES or SMILES.
    - Reference structures are *always* decoded from raw SELFIES.
    - Unextractable numerical predictions preserve official zero fallback,
      but are marked clearly, and extraction rate is separately reported.
    - Free-text captions use unchanged METEOR scoring after thinking removal.
    """
    from mol_postprocess import extract_chem, extract_numeric, remove_reasoning
    task = x['task']
    pred = remove_reasoning(str(x.get('prediction', '')))
    ans = str(x['reference'])
    if task in CHEM:
        import selfies as sf
        from rdkit import Chem, DataStructs
        extraction = extract_chem(pred)
        ptype, ptext = extraction.representation, extraction.value
        if not ptext:
            return {'score': 0.0, 'valid': False,
                    'parsed_prediction': None, 'parsed_reference': None,
                    'extraction_method': extraction.method,
                    'reason': extraction.reason or 'no_chemical_answer'}
        try:
            ref_smiles = sf.decoder(ans)
        except Exception:
            return {'score': 0.0, 'valid': False,
                    'parsed_prediction': ptext, 'parsed_reference': ans,
                    'extraction_method': extraction.method,
                    'reason': 'reference_selfies_decode_failed'}
        if ptype == 'SELFIES':
            try:
                pred_smiles = sf.decoder(ptext)
            except Exception:
                return {'score': 0.0, 'valid': False,
                        'parsed_prediction': ptext,
                        'parsed_reference': ref_smiles,
                        'extraction_method': extraction.method,
                        'reason': 'prediction_selfies_decode_failed'}
        else:
            pred_smiles = ptext
        mol1 = Chem.MolFromSmiles(pred_smiles)
        mol2 = Chem.MolFromSmiles(ref_smiles)
        if (mol1 is None or mol2 is None or
                mol1.GetNumAtoms() == 0 or mol2.GetNumAtoms() == 0):
            return {'score': 0.0, 'valid': False,
                    'parsed_prediction': pred_smiles, 'parsed_reference': ref_smiles,
                    'extraction_method': extraction.method,
                    'reason': 'invalid_or_empty_molecule'}
        fp1 = generator.GetFingerprint(mol1)
        fp2 = generator.GetFingerprint(mol2)
        return {'score': float(DataStructs.TanimotoSimilarity(fp1, fp2) * 100),
                'valid': True, 'parsed_prediction': pred_smiles,
                'parsed_reference': ref_smiles,
                'extraction_method': extraction.method}
    if task == 'PP':
        num, method, reason = extract_numeric(pred)
        # Same fallback as the original OpenCompass MAEEvaluator, but log it.
        value = num if num is not None else 0.0
        return {'score': abs(value - float(ans)),
                'parsed_prediction': value,
                'number_extracted': num is not None,
                'extraction_method': method,
                **({'reason': reason} if reason else {})}
    if task == 'MC':
        # METEOR evaluates the model-generated caption; do not mistakenly select
        # chemical fragments or numbers from its description.
        result = score_prediction(x, generator)
        result['extraction_method'] = 'full_caption_after_think_removal'
        return result
    raise ValueError(f'Unexpected task {task}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-path', default=Path('./test.jsonl'))
    p.add_argument('--predictions', default=Path('./eval_result/predictions.jsonl'))
    p.add_argument('--output-dir', default=Path('./eval_result'))
    p.add_argument('--allow-partial', action='store_true')
    p.add_argument('--max-samples-per-task', type=int, default=0)
    a = p.parse_args()
    data = read_test(a.data_path, a.max_samples_per_task)
    by_id = {x['id']: x for x in data}
    predictions = {}
    with open(a.predictions, encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            idx = row['id']
            if idx not in by_id:
                raise ValueError(f'Prediction has unknown ID: {idx}')
            if idx in predictions:
                raise ValueError(f'Duplicate prediction ID: {idx}')
            if row.get('reference') != by_id[idx]['reference']:
                raise ValueError(f'Reference mismatch at id={idx}')
            if row.get('task') != by_id[idx]['task']:
                raise ValueError(f'Task mismatch at id={idx}')
            predictions[idx] = row
    missing = sorted(by_id.keys() - predictions.keys())
    if missing and not a.allow_partial:
        raise ValueError(f'Missing {len(missing)} predictions; examples: {missing[:10]}. '
                         'Run inference to completion or pass --allow-partial.')
    import selfies  # noqa: F401; required for the 4 chemistry tasks
    from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.error')
    generator = GetMorganGenerator(radius=2, fpSize=2048)
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    task_scores = defaultdict(list)
    strict_scores = defaultdict(list)
    validity = Counter()
    extracted = Counter()
    extraction_methods = defaultdict(Counter)
    reason_counts = defaultdict(Counter)
    now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    with (out / f'scored_predictions_{now}.jsonl').open('w', encoding='utf-8') as dest:
        for idx in sorted(predictions):
            row = predictions[idx]
            measured = score_native_prediction(row, generator)
            strict = score_prediction(row, generator)
            task = row['task']
            task_scores[task].append(measured['score'])
            strict_scores[task].append(strict['score'])
            extraction_methods[task][measured.get('extraction_method', 'unknown')] += 1
            if measured.get('valid'):
                validity[task] += 1
            if measured.get('number_extracted'):
                extracted[task] += 1
            if measured.get('reason'):
                reason_counts[task][measured['reason']] += 1
            outrow = {**row, **measured, 'opencompass_strict': strict}
            dest.write(json.dumps(outrow, ensure_ascii=False) + '\n')
    results = []
    for task in TASK_ORDER:
        values = task_scores[task]
        score = (sum(values) / len(values)) if values else None
        full_size = sum(x['task'] == task for x in data)
        item = {'task': task, 'name': TASKS[task][0],
                'count': len(values), 'expected_count': full_size,
                'score': score,
                'opencompass_strict_score':
                    (sum(strict_scores[task]) / len(strict_scores[task]))
                    if strict_scores[task] else None,
                'metric': 'Morgan Tanimoto x 100' if task in CHEM else
                          'METEOR' if task == 'MC' else 'MAE',
                'better': 'lower' if task == 'PP' else 'higher',
                'valid_score': (100.0 * validity[task] / len(values)) if values and task in CHEM else None,
                'number_extraction_rate': (100.0 * extracted[task] / len(values)) if values and task == 'PP' else None,
                'failure_reasons': dict(reason_counts[task]),
                'extraction_methods': dict(extraction_methods[task]),
        }
        results.append(item)
    summary = {'dataset_size': len(data), 'num_predictions': len(predictions),
               'num_missing': len(missing),
               'score_protocol': 'Conservative extracted final answer, original task metrics; PP unextractable falls back to zero',
               'strict_score_protocol': 'OpenCompass-like parsing without SFT format adaptation (diagnostic only)',
               'overall': None,
               'overall_note': 'Not calculated: original metrics differ in scale and direction, and InternLM aggregate normalization is not public.',
               'metrics': results}
    (out / f'summary_{now}.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    with (out / f'summary_{now}.csv').open('w', newline='', encoding='utf-8') as f:
        columns = ['task', 'name', 'count', 'expected_count', 'metric', 'score',
                   'opencompass_strict_score', 'better', 'valid_score',
                   'number_extraction_rate']
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    print('Task  Count  Metric                   Native score    Strict score    Valid%')
    for r in results:
        val = f'{r["score"]:.4f}' if r['score'] is not None else 'N/A'
        strict = f'{r["opencompass_strict_score"]:.4f}' if r['opencompass_strict_score'] is not None else 'N/A'
        valid = f'{r["valid_score"]:.2f}' if r['valid_score'] is not None else '-'
        print(f'{r["task"]:4} {r["count"]:6}  {r["metric"]:23} {val:>12} {strict:>14} {valid:>9}')
    print(f'Saved {out / f"summary_{now}.json"}, {out / f"summary_{now}.csv"}, {out / f"scored_predictions_{now}.jsonl"}')
    if missing:
        print(f'WARNING: {len(missing)} predictions missing; use --allow-partial only for a smoke test')


if __name__ == '__main__':
    main()
