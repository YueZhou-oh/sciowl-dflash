"""Conservative postprocessor for Mol-Instructions generated responses.

This module never uses reference labels to identify an answer. It does not
perform chemistry scoring; the scorer does that after extraction.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict

THINK = re.compile(r'<think>.*?</think>', re.S | re.I)
TAG = re.compile(r'<(SELFIES|SMILES)>\s*(.*?)\s*</\1>', re.S | re.I)
SELFIES_FULL = re.compile(r'(?:\[[^\]\r\n]+\])+(?:\.(?:\[[^\]\r\n]+\])+)*')
NUMBER = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
NUMBER_RE = re.compile(NUMBER)
NUMBER_FULL = re.compile(rf'\s*({NUMBER})\s*')
NUMBER_BOX = re.compile(rf'(?:<NUMBER>\s*|\\boxed\{{)\s*({NUMBER})\s*(?:</NUMBER>|\}})',re.I)
NUMBER_WITH_UNIT = re.compile(rf'\s*({NUMBER})\s*(?:eV|kcal/mol|kJ/mol|Hartree|Ha)?\s*[.!]?\s*', re.I)
ALT_NUMBER_TAG = re.compile(rf'<MOLFORMULA>\s*({NUMBER})\s*</MOLFORMULA>', re.I)
ANSWER_PREFIX = re.compile(
    r'(?i)(?:\b(?:I believe the answer is|I would respond by saying|'
    r'I would say that|In my opinion, the answer would be|'
    r'From my perspective, the answer is|My response is|'
    r'As for my answer, it is|I would conclude that|In response, I offer|'
    r'The solution I propose is|The answer is|The final answer is|'
    r'Final answer|Predicted (?:product|reagents?|reactants?))\s*[:：]?\s*)'
)
LABELED_STRUCTURE = re.compile(r'(?im)^\s*\*{0,2}(SELFIES|SMILES)\*{0,2}\s*[:：=]\s*(.+?)\s*$')
FENCE = re.compile(r'```\s*(selfies|smiles)\s*\n(.*?)\n```', re.I | re.S)


@dataclass
class Extraction:
    value: str | None
    representation: str | None
    method: str
    reason: str | None = None

    def to_dict(self):
        return asdict(self)


def remove_reasoning(text: str) -> str:
    """Remove explicit Qwen3 reasoning tags; do not drop untagged prose."""
    text = str(text or '')
    if '<think>' not in text.lower() and '</think>' in text.lower():
        return re.split(r'</think>', text, flags=re.I)[-1].strip()
    text = THINK.sub('', text).strip()
    # Incomplete thinking segments should never be interpreted as final answers.
    if re.search(r'<think>', text, re.I):
        text = re.split(r'<think>', text, maxsplit=1, flags=re.I)[0].strip()
    return text


def normalize_candidate(text: str) -> str:
    text = text.strip().strip('`').strip().strip('"\'').strip()
    text = re.sub(r'^(?:\*\*|\*|\s)+', '', text).strip()
    text = re.sub(r'(?:\*\*|\*)+$', '', text).strip()
    text = text.strip().rstrip(';').strip()
    return text


def detect_chem_form(candidate: str, hinted: str | None = None) -> Extraction:
    """Reject uncertain prose rather than auto-extracting guessed substrings."""
    candidate = normalize_candidate(candidate)
    # A literal final period can be ambiguous with a SMILES disconnected atom
    # delimiter. We only trim a natural-language sentence-ending period when
    # it is followed by whitespace; the candidate parser gets the rest intact.
    if not candidate:
        return Extraction(None, None, 'none', 'empty_candidate')
    if any(c.isspace() for c in candidate):
        return Extraction(None, None, 'none', 'candidate_contains_prose_or_spaces')
    if re.search(r'[<>`{}]',candidate):
        return Extraction(None,None,'none','candidate_contains_markup')
    if hinted:
        return Extraction(candidate, hinted.upper(), 'explicit_format')
    if SELFIES_FULL.fullmatch(candidate):
        return Extraction(candidate, 'SELFIES', 'raw_selfies')
    # Basic lexicographic filter, followed by authoritative RDKit validity in scorer.
    # Avoid selecting common words that can accidentally look like atoms.
    if not re.fullmatch(r'[A-Za-z0-9@+\-\[\]().=#/$\\:%*]+', candidate):
        return Extraction(None,None,'none','not_chemical_syntax')
    if len(candidate) > 4096:
        return Extraction(None,None,'none','overlong_candidate')
    if not re.search(r'[0-9@+\-\[\]().=#/\\]',candidate):
        if candidate not in {'C','N','O','S','P','F','Cl','Br','I','CO','CN','CC','NO','OC','NC'}:
            return Extraction(None,None,'none','ambiguous_plain_word')
    return Extraction(candidate, 'SMILES', 'raw_smiles')


def extract_chem(text: str) -> Extraction:
    text = remove_reasoning(text)
    if not text:
        return Extraction(None,None,'none','empty_final_response')
    tags = list(TAG.finditer(text))
    if tags:
        # The official evaluator selects last valid tag. In a verbose rationale,
        # multiple tagged species can be reagents/intermediates; flag for review.
        if len(tags) > 1 and not ANSWER_PREFIX.search(text[tags[-2].end():tags[-1].start()]):
            return Extraction(None,None,'none','multiple_tagged_molecules_ambiguous')
        m = tags[-1]
        return Extraction(normalize_candidate(m.group(2)),m.group(1).upper(),
                          'tagged_last' if len(tags)==1 else 'tagged_last_of_many')

    # Plain-format label such as 'SMILES: CC(O)C'; inspect the last such label.
    labeled = list(LABELED_STRUCTURE.finditer(text))
    if labeled:
        m=labeled[-1]
        candidate = m.group(2).splitlines()[0]
        result = detect_chem_form(candidate, m.group(1).upper())
        result.method = 'labeled_format'
        return result
    fences=list(FENCE.finditer(text))
    if len(fences)==1:
        m=fences[0]
        result = detect_chem_form(m.group(2), m.group(1).upper())
        result.method='typed_code_fence'
        return result
    if len(fences)>1:
        return Extraction(None,None,'none','multiple_code_fences_ambiguous')

    result = detect_chem_form(text)
    if result.value is not None:
        return result

    # Only use untagged text after explicit final-answer phrase. If multiple
    # candidate-bearing answer statements appear, conservatively take the last.
    prefixes=list(ANSWER_PREFIX.finditer(text))
    if prefixes:
        tail=text[prefixes[-1].end():].strip()
        # One line, not the whole narrative beyond the proposed answer.
        if not tail:
            return Extraction(
                None,
                None,
                'answer_phrase',
                'empty_answer_after_prefix',
            )

        first = tail.splitlines()[0].strip()
        first = first.split('  ')[0].strip()
        # Remove sentence punctuation *only* when not a legitimate SMILES dot.
        first = re.sub(r'\s+[.!]\s*$', '', first)
        result = detect_chem_form(first)
        result.method='answer_phrase'
        return result
    return Extraction(None,None,'none','no_unambiguous_molecular_answer')


def extract_numeric(text: str) -> tuple[float | None, str, str | None]:
    text=remove_reasoning(text)
    matches=list(NUMBER_BOX.finditer(text))
    if matches:
        return float(matches[-1].group(1)), 'boxed_or_number_tag', None
    alt=list(ALT_NUMBER_TAG.finditer(text))
    if alt:
        return float(alt[-1].group(1)), 'molformula_numeric_tag', None
    m=NUMBER_WITH_UNIT.fullmatch(text)
    if m:
        return float(m.group(1)), 'standalone_numeric', None
    prefixes=list(ANSWER_PREFIX.finditer(text))
    if prefixes:
        # tail=text[prefixes[-1].end():].strip().splitlines()[0]
        # m=NUMBER_WITH_UNIT.fullmatch(tail)
        tail = text[prefixes[-1].end():].strip()
        if not tail:
            return (
                None,
                'answer_phrase_numeric',
                'empty_answer_after_prefix',
            )

        tail = tail.splitlines()[0]
        m = NUMBER_WITH_UNIT.fullmatch(tail)
        if m:
            return float(m.group(1)), 'answer_phrase_numeric', None
    return None,'none','no_unambiguous_numeric_answer'
