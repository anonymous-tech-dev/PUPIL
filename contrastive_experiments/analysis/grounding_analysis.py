"""
Grounding Analysis: Vanilla SFT vs Contrastive SFT
====================================================
Tests the hypothesis that vanilla SFT learns surface-level response patterns
(gaming BLEU scores) while failing on the key visual detail, and that
contrastive SFT improves visual grounding.

Core idea: CGBench questions are MCQ-based. The ground-truth answer references
a specific "key entity" from the video (a number, color, name, text, object,
timestamp, etc.). We extract these key entities from the reference and check
whether the prediction preserves them correctly vs. substituting a wrong one
in the same semantic slot.

Non-LLM Metrics Introduced
---------------------------
1. **Key-Entity Recall (KER)**: For each sample, extract named entities,
   numbers, and quoted/specific terms from the reference. Measure what
   fraction appear in the prediction. Aggregated per sub_category.

2. **Slot-Error Rate (SER)**: Among samples where the prediction contains a
   *different* entity in the same slot (e.g. "119" instead of "12"), count
   the rate of such substitutions. This is the "pattern gaming" signal.

3. **Structural Similarity sans Entities (SSSE)**: Measure how similar the
   prediction is to the reference *after removing* key entities. High SSSE +
   low KER = model copies the template but not the substance.

4. **Grounding Gap** = SSSE - KER: positive means model is better at copying
   structure than substance → evidence of pattern gaming.
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from difflib import SequenceMatcher

import numpy as np

# ─── Entity extraction ───────────────────────────────────────────────────────

# Patterns for visually-grounded entities
_NUMBER_RE = re.compile(r'\b\d[\d,\.]*\b')
_QUOTED_RE = re.compile(r'["\u201c\u201d](.*?)["\u201c\u201d]')
_TIME_RE = re.compile(r'\b\d{1,2}:\d{2}(?::\d{2})?\b')
_COLOR_RE = re.compile(
    r'\b(red|blue|green|yellow|orange|purple|pink|black|white|brown|gray|grey'
    r'|cyan|magenta|golden|silver|beige|maroon|navy|teal|violet|indigo)\b',
    re.IGNORECASE,
)
# Capitalized multi-word proper nouns / named entities (simple heuristic)
_PROPER_NOUN_RE = re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b')

# Specific unit patterns (yen, dollars, kg, cm, etc.)
_UNIT_NUMBER_RE = re.compile(
    r'\b(\d[\d,\.]*)\s*(yen|dollar|dollars|\$|€|£|%|kg|cm|mm|m|km|lbs?|oz'
    r'|ml|liters?|minutes?|seconds?|hours?|mph|km/h|points?|goals?)\b',
    re.IGNORECASE,
)


def extract_key_entities(text: str) -> set[str]:
    """Extract visually-grounded key entities from text."""
    entities = set()

    # Numbers with units (keep as combined)
    for m in _UNIT_NUMBER_RE.finditer(text):
        entities.add(m.group(0).lower().strip())

    # Standalone numbers
    for m in _NUMBER_RE.finditer(text):
        entities.add(m.group(0))

    # Times
    for m in _TIME_RE.finditer(text):
        entities.add(m.group(0))

    # Colors
    for m in _COLOR_RE.finditer(text):
        entities.add(m.group(0).lower())

    # Quoted terms
    for m in _QUOTED_RE.finditer(text):
        entities.add(m.group(1).strip().lower())

    # Proper nouns (only keep those with 2+ chars)
    for m in _PROPER_NOUN_RE.finditer(text):
        if len(m.group(0)) > 2:
            entities.add(m.group(0))

    return entities


def normalize_number(s: str) -> str:
    """Normalize number strings for comparison: remove commas, trailing .0"""
    s = s.replace(',', '')
    try:
        f = float(s)
        if f == int(f):
            return str(int(f))
        return f"{f:.4f}".rstrip('0').rstrip('.')
    except ValueError:
        return s.lower()


def entity_match(ref_ent: str, pred_ents: set[str]) -> bool:
    """Check if a reference entity is matched in prediction entities."""
    # Exact
    if ref_ent in pred_ents:
        return True
    # Normalized number match
    ref_norm = normalize_number(ref_ent)
    for p in pred_ents:
        if normalize_number(p) == ref_norm:
            return True
    # Case-insensitive
    ref_low = ref_ent.lower()
    for p in pred_ents:
        if p.lower() == ref_low:
            return True
    return False


# ─── Structural similarity sans entities ──────────────────────────────────────

def remove_entities(text: str, entities: set[str]) -> str:
    """Remove all entity mentions from text, leaving structural template."""
    result = text
    # Sort by length descending to avoid partial replacements
    for ent in sorted(entities, key=len, reverse=True):
        result = result.replace(ent, ' _ENT_ ')
    # Also strip numbers and proper nouns generically
    result = _NUMBER_RE.sub(' _NUM_ ', result)
    result = _PROPER_NOUN_RE.sub(' _NE_ ', result)
    # Collapse whitespace
    result = re.sub(r'\s+', ' ', result).strip()
    return result


def structural_similarity(ref: str, pred: str, ref_ents: set[str], pred_ents: set[str]) -> float:
    """Compute text similarity after stripping entities."""
    ref_clean = remove_entities(ref, ref_ents)
    pred_clean = remove_entities(pred, pred_ents)
    return SequenceMatcher(None, ref_clean, pred_clean).ratio()


# ─── Slot error detection ────────────────────────────────────────────────────

def detect_slot_errors(ref: str, pred: str, ref_ents: set[str], pred_ents: set[str]) -> list[dict]:
    """
    Detect cases where prediction has a *different* entity in the same slot.
    Returns list of {ref_entity, pred_entity, slot_type} dicts.
    """
    errors = []

    # Check numbers: find numbers in ref that are NOT in pred, but pred has
    # a different number nearby in the sentence structure
    ref_nums = set(_NUMBER_RE.findall(ref))
    pred_nums = set(_NUMBER_RE.findall(pred))

    for rn in ref_nums:
        if not entity_match(rn, pred_nums):
            # Ref has a number not in prediction - is there a substitution?
            # Check if pred has a different number in a similar context
            for pn in pred_nums:
                if normalize_number(pn) != normalize_number(rn):
                    # Check if they appear in similar surrounding context
                    ref_ctx = _get_number_context(ref, rn)
                    pred_ctx = _get_number_context(pred, pn)
                    if ref_ctx and pred_ctx and SequenceMatcher(None, ref_ctx, pred_ctx).ratio() > 0.5:
                        errors.append({
                            'ref_entity': rn,
                            'pred_entity': pn,
                            'slot_type': 'number',
                        })
                        break

    # Check named entities
    ref_only = {e for e in ref_ents if not entity_match(e, pred_ents)}
    pred_only = {e for e in pred_ents if not entity_match(e, ref_ents)}

    for re_ in ref_only:
        for pe in pred_only:
            # Same type heuristic: both are proper nouns, both are colors, etc.
            if (_is_number_like(re_) and _is_number_like(pe)):
                continue  # already handled above
            if (_COLOR_RE.fullmatch(re_) and _COLOR_RE.fullmatch(pe)):
                errors.append({'ref_entity': re_, 'pred_entity': pe, 'slot_type': 'color'})
            elif (re_[0].isupper() and pe[0].isupper() and
                  not _is_number_like(re_) and not _is_number_like(pe)):
                errors.append({'ref_entity': re_, 'pred_entity': pe, 'slot_type': 'named_entity'})

    return errors


def _get_number_context(text: str, number: str, window: int = 30) -> str:
    """Get surrounding context of a number in text."""
    idx = text.find(number)
    if idx < 0:
        return ""
    start = max(0, idx - window)
    end = min(len(text), idx + len(number) + window)
    ctx = text[start:end]
    # Replace the number itself
    ctx = ctx.replace(number, '_X_', 1)
    return ctx.lower()


def _is_number_like(s: str) -> bool:
    return bool(_NUMBER_RE.fullmatch(s))


# ─── Main analysis ───────────────────────────────────────────────────────────

def analyze_predictions(predictions: list[dict]) -> dict:
    """Compute grounding metrics for a set of predictions."""
    results = []

    for item in predictions:
        ref = item['reference']
        pred = item['prediction']
        sub_cat = item.get('metadata', {}).get('sub_category', 'unknown')

        ref_ents = extract_key_entities(ref)
        pred_ents = extract_key_entities(pred)

        if not ref_ents:
            continue  # skip if no extractable entities in reference

        # Key-Entity Recall
        matched = sum(1 for e in ref_ents if entity_match(e, pred_ents))
        ker = matched / len(ref_ents)

        # Structural Similarity sans Entities
        ssse = structural_similarity(ref, pred, ref_ents, pred_ents)

        # Slot errors
        slot_errors = detect_slot_errors(ref, pred, ref_ents, pred_ents)

        # Grounding gap
        grounding_gap = ssse - ker

        results.append({
            'id': item['id'],
            'sub_category': sub_cat,
            'ker': ker,
            'ssse': ssse,
            'grounding_gap': grounding_gap,
            'n_ref_entities': len(ref_ents),
            'n_matched': matched,
            'n_slot_errors': len(slot_errors),
            'slot_errors': slot_errors,
            'bleu_4': item.get('metrics', {}).get('bleu_4', None),
            'rouge_l': item.get('metrics', {}).get('rouge_l', None),
        })

    return results


def aggregate(results: list[dict]) -> dict:
    """Aggregate results overall and by sub_category."""
    by_cat = defaultdict(list)
    for r in results:
        by_cat[r['sub_category']].append(r)
        by_cat['__overall__'].append(r)

    agg = {}
    for cat, items in sorted(by_cat.items()):
        n = len(items)
        agg[cat] = {
            'n': n,
            'ker_mean': np.mean([i['ker'] for i in items]),
            'ssse_mean': np.mean([i['ssse'] for i in items]),
            'grounding_gap_mean': np.mean([i['grounding_gap'] for i in items]),
            'slot_error_rate': np.mean([1 if i['n_slot_errors'] > 0 else 0 for i in items]),
            'avg_slot_errors': np.mean([i['n_slot_errors'] for i in items]),
            'bleu_4_mean': np.mean([i['bleu_4'] for i in items if i['bleu_4'] is not None]),
            'rouge_l_mean': np.mean([i['rouge_l'] for i in items if i['rouge_l'] is not None]),
        }
    return agg


def print_comparison(vanilla_agg, contrastive_agg, vanilla_results, contrastive_results):
    """Print a formatted comparison table."""

    print("=" * 110)
    print("GROUNDING ANALYSIS: Vanilla SFT vs Contrastive SFT (V-04)")
    print("=" * 110)

    # Overall comparison
    cats = sorted(set(list(vanilla_agg.keys()) + list(contrastive_agg.keys())))
    cats = [c for c in cats if c != '__overall__']
    cats = ['__overall__'] + cats

    header = f"{'Category':<25} │ {'N':>4} │ {'KER↑':>7} {'SSSE':>7} {'Gap↓':>7} {'SER↓':>7} │ {'KER↑':>7} {'SSSE':>7} {'Gap↓':>7} {'SER↓':>7} │ {'ΔKER':>7} {'ΔSER':>7}"
    print(f"\n{'':>25}   {'':>4}   {'── Vanilla SFT ──':^30}   {'── Contrastive ──':^30}   {'─ Delta ─':^15}")
    print(header)
    print("─" * 110)

    for cat in cats:
        label = 'OVERALL' if cat == '__overall__' else cat
        v = vanilla_agg.get(cat, {})
        c = contrastive_agg.get(cat, {})
        if not v or not c:
            continue

        d_ker = c['ker_mean'] - v['ker_mean']
        d_ser = c['slot_error_rate'] - v['slot_error_rate']

        print(f"{label:<25} │ {v['n']:>4} │ "
              f"{v['ker_mean']:>6.1%} {v['ssse_mean']:>6.1%} {v['grounding_gap_mean']:>+6.1%} {v['slot_error_rate']:>6.1%} │ "
              f"{c['ker_mean']:>6.1%} {c['ssse_mean']:>6.1%} {c['grounding_gap_mean']:>+6.1%} {c['slot_error_rate']:>6.1%} │ "
              f"{d_ker:>+6.1%} {d_ser:>+6.1%}")

    # Show worst slot errors from vanilla
    print("\n" + "=" * 110)
    print("EXAMPLE SLOT ERRORS (Vanilla SFT) — model copies template but gets key detail wrong")
    print("=" * 110)

    errors_v = [(r, e) for r in vanilla_results for e in r['slot_errors']]
    # Find corresponding contrastive result
    contrastive_by_id = {r['id']: r for r in contrastive_results}

    shown = 0
    for r, err in sorted(errors_v, key=lambda x: -x[0]['grounding_gap'])[:15]:
        if shown >= 15:
            break
        c_r = contrastive_by_id.get(r['id'])
        c_has_error = c_r and c_r['n_slot_errors'] > 0 if c_r else '?'

        # Get the actual ref and pred from the predictions
        print(f"\n  [{r['id']}] ({r['sub_category']})")
        print(f"    Slot type: {err['slot_type']} | Ref: '{err['ref_entity']}' → Pred: '{err['pred_entity']}'")
        print(f"    Vanilla:  KER={r['ker']:.2f}  SSSE={r['ssse']:.2f}  Gap={r['grounding_gap']:+.2f}")
        if c_r:
            print(f"    Contrast: KER={c_r['ker']:.2f}  SSSE={c_r['ssse']:.2f}  Gap={c_r['grounding_gap']:+.2f}  "
                  f"SlotErr={'YES' if c_has_error else 'NO'}")
        shown += 1

    # Summary statistics
    print("\n" + "=" * 110)
    print("SUMMARY")
    print("=" * 110)
    vo = vanilla_agg['__overall__']
    co = contrastive_agg['__overall__']
    print(f"  Vanilla   — KER: {vo['ker_mean']:.1%}  |  Slot Error Rate: {vo['slot_error_rate']:.1%}  |  "
          f"Grounding Gap: {vo['grounding_gap_mean']:+.3f}  |  BLEU-4: {vo['bleu_4_mean']:.3f}")
    print(f"  Contrastv — KER: {co['ker_mean']:.1%}  |  Slot Error Rate: {co['slot_error_rate']:.1%}  |  "
          f"Grounding Gap: {co['grounding_gap_mean']:+.3f}  |  BLEU-4: {co['bleu_4_mean']:.3f}")

    d_ker = co['ker_mean'] - vo['ker_mean']
    d_ser = co['slot_error_rate'] - vo['slot_error_rate']
    print(f"\n  ΔKER (contrastive - vanilla): {d_ker:+.1%}  {'✓ BETTER' if d_ker > 0 else '✗ WORSE'}")
    print(f"  ΔSER (contrastive - vanilla): {d_ser:+.1%}  {'✓ BETTER' if d_ser < 0 else '✗ WORSE'}")

    gap_delta = co['grounding_gap_mean'] - vo['grounding_gap_mean']
    print(f"  ΔGap (contrastive - vanilla): {gap_delta:+.3f}  {'✓ BETTER' if gap_delta < 0 else '✗ WORSE'}")


def main():
    vanilla_path = Path('/workspace/Pupil/contrastive_experiments/outputs/'
                        'vanilla_sft_fps1_lr2e-5_ep1_65536seq/test_results_matched/predictions.json')
    contrastive_path = Path('/workspace/Pupil/contrastive_experiments/outputs/'
                            'V-04_generative_fps1_lambda0.4_alpha1.0_lr2e-5_ep1_65536seq/'
                            'test_results_matched/predictions.json')

    vanilla_data = json.load(open(vanilla_path))
    contrastive_data = json.load(open(contrastive_path))

    print(f"Loaded {len(vanilla_data)} vanilla, {len(contrastive_data)} contrastive predictions\n")

    vanilla_results = analyze_predictions(vanilla_data)
    contrastive_results = analyze_predictions(contrastive_data)

    print(f"Analyzed {len(vanilla_results)} vanilla, {len(contrastive_results)} contrastive "
          f"(samples with extractable entities)\n")

    vanilla_agg = aggregate(vanilla_results)
    contrastive_agg = aggregate(contrastive_results)

    print_comparison(vanilla_agg, contrastive_agg, vanilla_results, contrastive_results)

    # Save detailed results
    out_dir = Path('/workspace/Pupil/contrastive_experiments/analysis')
    out_dir.mkdir(exist_ok=True)
    json.dump(vanilla_results, open(out_dir / 'vanilla_grounding_detail.json', 'w'), indent=2, default=str)
    json.dump(contrastive_results, open(out_dir / 'contrastive_grounding_detail.json', 'w'), indent=2, default=str)
    print(f"\nDetailed results saved to {out_dir}/")


if __name__ == '__main__':
    main()
