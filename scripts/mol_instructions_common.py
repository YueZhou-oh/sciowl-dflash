"""Mol-Instructions original SFT data loader and Qwen3 no-thinking prompts.

For inference, we use only the original dataset fields:
    USER = instruction + "\n" + input
There are no OpenCompass system messages, task hints or Question/Answer labels.
Scores can still be calculated by the separate evaluation script.
"""
import json
from pathlib import Path

TASKS = {
    'RP': ('reagent prediction', 'reagent_prediction'),
    'MG': ('description-guided molecule design', 'description_guided_molecule_design'),
    'FS': ('forward reaction prediction', 'forward_reaction_prediction'),
    'RS': ('retrosynthesis', 'retrosynthesis'),
    'PP': ('property prediction', 'property_prediction'),
    'MC': ('molecular description generation', 'molecular_description_generation'),
}
TASK_BY_NAME = {name: code for code, (name, _) in TASKS.items()}
TASK_ORDER = ('FS', 'MC', 'MG', 'PP', 'RP', 'RS')

def read_test(path, limit_per_task=0):
    """Read original .jsonl or JSON array, preserving input line numbers for resume."""
    path = Path(path)
    if path.suffix.lower() == '.jsonl':
        with path.open(encoding='utf-8') as f:
            data = (json.loads(line) for line in f if line.strip())
            items = list(data)
    else:
        with path.open(encoding='utf-8') as f:
            items = json.load(f)
        if isinstance(items, dict):
            if 'test' not in items or not isinstance(items['test'], list):
                raise ValueError('JSON object must have a list-valued "test" key')
            items = items['test']
        if not isinstance(items, list):
            raise ValueError('Expected JSON array or JSONL')
    counts = {t: 0 for t in TASKS}
    results = []
    for index, x in enumerate(items):
        if not isinstance(x, dict):
            raise ValueError(f'Item {index} is not an object')
        meta = x.get('metadata', {})
        if isinstance(meta, str):
            meta = json.loads(meta)
        name = meta.get('task') or str(x.get('source_file', '')).split('/')[-1].replace('.json', '').replace('_', ' ')
        code = TASK_BY_NAME.get(name)
        if code is None:
            raise ValueError(f'Unknown task at index {index}: {name}')
        if limit_per_task and counts[code] >= limit_per_task:
            continue
        if not isinstance(x.get('input'), str) or not isinstance(x.get('output'), (str, int, float)):
            raise ValueError(f'Invalid input/output at index {index}')
        counts[code] += 1
        results.append({
            'id': index, 'task': code, 'instruction': str(x.get('instruction', '')),
            'input': x['input'], 'reference': str(x['output']),
            'source_file': str(x.get('source_file', '')),
            'source_index': x.get('source_index'),
        })
    return results


def create_messages(x):
    """Build the original SFT user content without added benchmark instructions."""
    instruction = x['instruction'].strip()
    user_input = x['input'].strip()
    content = '\n'.join(part for part in (instruction, user_input) if part)
    if not content:
        raise ValueError('Empty instruction and input in a test sample')
    return [{'role': 'user', 'content': content}]


def create_prompt(x, tokenizer):
    """Apply the Qwen3 chat template once, with thinking explicitly disabled.

    Fail rather than silently ignoring enable_thinking=False on incompatible
    tokenizer implementations. No additional role or output-format hints.
    """
    if tokenizer is None:
        raise ValueError('Qwen3 tokenizer is required')
    messages = create_messages(x)
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            'Tokenizer does not support explicit Qwen3 enable_thinking=False; '
            'update the tokenizer/transformers instead of silently enabling thinking.'
        ) from exc
    if not isinstance(prompt, str) or not prompt:
        raise RuntimeError('Tokenizer produced an empty/non-string Qwen3 prompt')
    # Incomplete reasoning preamble means the model may continue generating
    # analysis instead of the final answer (and thus thinking was not disabled).
    if prompt.count('<think>') > prompt.count('</think>'):
        raise RuntimeError('Qwen3 prompt contains an open <think> tag')
    return prompt
