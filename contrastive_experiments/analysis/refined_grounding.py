"""
Refined Grounding Analysis — Key Visual Detail Accuracy
========================================================
Focuses on the core hypothesis: does the model get the ONE key visual detail
right (number, color, object name, text on screen) or does it copy the
sentence template and hallucinate the detail?

Metrics:
  1. Template Match Rate (TMR): Does prediction use similar sentence structure?
  2. Key Detail Match (KDM): Does prediction get the specific visual detail right?
  3. Gaming Score = TMR - KDM: High positive = copying template, wrong detail
  4. Per-sample pairwise comparison: vanilla vs contrastive, who gets more right?
"""

import json, re, sys
from collections import defaultdict, Counter
from pathlib import Path
from difflib import SequenceMatcher
import numpy as np

# ─── Load data ────────────────────────────────────────────────────────────────

VANILLA_PATH = Path('outputs/vanilla_sft_fps1_lr2e-5_ep1_65536seq/test_results_matched/predictions.json')
CONTRASTIVE_PATH = Path('outputs/V-04_generative_fps1_lambda0.4_alpha1.0_lr2e-5_ep1_65536seq/test_results_matched/predictions.json')

vanilla = json.load(open(VANILLA_PATH))
contrastive = json.load(open(CONTRASTIVE_PATH))
v_by_id = {r['id']: r for r in vanilla}
c_by_id = {r['id']: r for r in contrastive}

# ─── Extract key visual details ──────────────────────────────────────────────

NUM_RE = re.compile(r'\b\d[\d,\.]*\b')
TIME_RE = re.compile(r'\b\d{1,2}:\d{2}(?::\d{2})?\b')
COLOR_RE = re.compile(
    r'\b(red|blue|green|yellow|orange|purple|pink|black|white|brown|gray|grey'
    r'|cyan|magenta|golden|silver|beige|maroon|navy|teal|violet|indigo'
    r'|rose-red|off-white)\b', re.I)

# Categories where the key visual detail is most diagnosable
VISUAL_DETAIL_CATS = {
    'Text Perception', 'Text Cognition',
    'Time Perception', 'Time Cognition',
    'Entity Perception', '2D Spatial Perception',
    'Scene Perception',
}


def extract_numbers(text):
    """Extract all standalone numbers."""
    # First extract times so we don't double-count
    times = set(TIME_RE.findall(text))
    nums = set(NUM_RE.findall(text))
    return nums, times


def extract_colors(text):
    return set(m.group(0).lower() for m in COLOR_RE.finditer(text))


def normalize_num(s):
    s = s.replace(',', '')
    try:
        f = float(s)
        return int(f) if f == int(f) else round(f, 4)
    except:
        return s


def nums_match(ref_nums, pred_nums):
    """Check if all reference numbers appear in prediction."""
    ref_norm = {normalize_num(n) for n in ref_nums}
    pred_norm = {normalize_num(n) for n in pred_nums}
    if not ref_norm:
        return None  # no numbers to check
    return ref_norm.issubset(pred_norm)


def colors_match(ref_colors, pred_colors):
    if not ref_colors:
        return None
    return ref_colors.issubset(pred_colors)


def template_similarity(ref, pred):
    """Similarity after replacing all numbers/colors with placeholders."""
    def neutralize(t):
        t = NUM_RE.sub('_N_', t)
        t = TIME_RE.sub('_T_', t)
        t = COLOR_RE.sub('_C_', t)
        t = re.sub(r'\s+', ' ', t).strip().lower()
        return t
    return SequenceMatcher(None, neutralize(ref), neutralize(pred)).ratio()


# ─── Analyze each sample ─────────────────────────────────────────────────────

results = []

for v_item in vanilla:
    id_ = v_item['id']
    c_item = c_by_id.get(id_)
    if not c_item:
        continue

    cat = v_item['metadata'].get('sub_category', '')
    ref = v_item['reference']
    v_pred = v_item['prediction']
    c_pred = c_item['prediction']

    ref_nums, ref_times = extract_numbers(ref)
    ref_colors = extract_colors(ref)

    v_nums, v_times = extract_numbers(v_pred)
    v_colors = extract_colors(v_pred)

    c_nums, c_times = extract_numbers(c_pred)
    c_colors = extract_colors(c_pred)

    # Number match
    all_ref_nums = ref_nums | ref_times
    v_num_ok = nums_match(all_ref_nums, v_nums | v_times)
    c_num_ok = nums_match(all_ref_nums, c_nums | c_times)

    # Color match
    v_col_ok = colors_match(ref_colors, v_colors)
    c_col_ok = colors_match(ref_colors, c_colors)

    # Template similarity
    v_tmpl = template_similarity(ref, v_pred)
    c_tmpl = template_similarity(ref, c_pred)

    # Only include if there's something to measure
    has_detail = (v_num_ok is not None) or (v_col_ok is not None)
    if not has_detail:
        continue

    # Composite key detail match (all details correct)
    checks_v = [x for x in [v_num_ok, v_col_ok] if x is not None]
    checks_c = [x for x in [c_num_ok, c_col_ok] if x is not None]
    v_detail_ok = all(checks_v) if checks_v else None
    c_detail_ok = all(checks_c) if checks_c else None

    results.append({
        'id': id_, 'cat': cat, 'ref': ref,
        'v_pred': v_pred, 'c_pred': c_pred,
        'v_num_ok': v_num_ok, 'c_num_ok': c_num_ok,
        'v_col_ok': v_col_ok, 'c_col_ok': c_col_ok,
        'v_detail_ok': v_detail_ok, 'c_detail_ok': c_detail_ok,
        'v_tmpl': v_tmpl, 'c_tmpl': c_tmpl,
        'v_bleu4': v_item['metrics']['bleu_4'],
        'c_bleu4': c_item['metrics']['bleu_4'],
        'ref_nums': all_ref_nums, 'ref_colors': ref_colors,
        'v_nums': v_nums | v_times, 'c_nums': c_nums | c_times,
    })

# ─── Aggregate ────────────────────────────────────────────────────────────────

print(f"Samples with measurable visual details: {len(results)}")
print()

# Overall + per-category
def report(items, label):
    n = len(items)
    if n == 0:
        return

    v_detail = [r['v_detail_ok'] for r in items if r['v_detail_ok'] is not None]
    c_detail = [r['c_detail_ok'] for r in items if r['c_detail_ok'] is not None]
    v_tmpl = [r['v_tmpl'] for r in items]
    c_tmpl = [r['c_tmpl'] for r in items]

    v_kdm = np.mean(v_detail) if v_detail else float('nan')
    c_kdm = np.mean(c_detail) if c_detail else float('nan')
    v_tmr = np.mean(v_tmpl)
    c_tmr = np.mean(c_tmpl)
    v_game = v_tmr - v_kdm if not np.isnan(v_kdm) else float('nan')
    c_game = c_tmr - c_kdm if not np.isnan(c_kdm) else float('nan')

    v_bleu = np.mean([r['v_bleu4'] for r in items])
    c_bleu = np.mean([r['c_bleu4'] for r in items])

    # Pairwise: how often does contrastive get the detail right when vanilla doesn't?
    c_wins = sum(1 for r in items if r['c_detail_ok'] and not r['v_detail_ok'])
    v_wins = sum(1 for r in items if r['v_detail_ok'] and not r['c_detail_ok'])
    both_right = sum(1 for r in items if r['v_detail_ok'] and r['c_detail_ok'])
    both_wrong = sum(1 for r in items if r['v_detail_ok'] is not None and not r['v_detail_ok']
                     and r['c_detail_ok'] is not None and not r['c_detail_ok'])

    print(f"  {label:<28} n={n:>4}  │ "
          f"KDM: V={v_kdm:>5.1%} C={c_kdm:>5.1%} (Δ={c_kdm-v_kdm:>+5.1%}) │ "
          f"TMR: V={v_tmr:>5.1%} C={c_tmr:>5.1%} │ "
          f"Gaming: V={v_game:>+5.1%} C={c_game:>+5.1%} │ "
          f"BLEU4: V={v_bleu:.3f} C={c_bleu:.3f}")
    print(f"  {'':28}        │ "
          f"Pairwise: C-wins={c_wins} V-wins={v_wins} Both✓={both_right} Both✗={both_wrong}")

by_cat = defaultdict(list)
for r in results:
    by_cat[r['cat']].append(r)
    by_cat['__OVERALL__'].append(r)

print("=" * 130)
print("KEY VISUAL DETAIL ACCURACY — Vanilla SFT vs Contrastive SFT")
print("  KDM = Key Detail Match (are the exact numbers/colors/entities correct?)")
print("  TMR = Template Match Rate (structural similarity ignoring details)")
print("  Gaming Score = TMR - KDM (high positive = copies template but wrong details)")
print("=" * 130)

report(results, 'OVERALL')
print("─" * 130)
for cat in sorted(by_cat.keys()):
    if cat == '__OVERALL__':
        continue
    report(by_cat[cat], cat)

# ─── Show "pattern gaming" examples ──────────────────────────────────────────

print("\n" + "=" * 130)
print("TOP PATTERN-GAMING EXAMPLES: High template match + wrong key detail (Vanilla)")
print("=" * 130)

gaming_examples = [r for r in results if r['v_detail_ok'] == False and r['v_tmpl'] > 0.6]
gaming_examples.sort(key=lambda r: r['v_tmpl'], reverse=True)

for r in gaming_examples[:20]:
    c_status = '✓' if r['c_detail_ok'] else '✗'
    print(f"\n  [{r['id']}] ({r['cat']})  TMR={r['v_tmpl']:.2f}")
    print(f"    REF: {r['ref'][:150]}")
    print(f"    VAN: {r['v_pred'][:150]}")
    print(f"    CON: {r['c_pred'][:150]} [{c_status}]")
    if r['ref_nums']:
        print(f"    Numbers — ref:{r['ref_nums']}  van:{r['v_nums']}  con:{r['c_nums']}")
    if r['ref_colors']:
        print(f"    Colors  — ref:{r['ref_colors']}  van:{extract_colors(r['v_pred'])}  con:{extract_colors(r['c_pred'])}")

# ─── Specific number/color accuracy breakdown ────────────────────────────────

print("\n" + "=" * 130)
print("NUMBER ACCURACY (samples with numbers in reference)")
print("=" * 130)

num_items = [r for r in results if r['v_num_ok'] is not None]
v_num_acc = np.mean([r['v_num_ok'] for r in num_items])
c_num_acc = np.mean([r['c_num_ok'] for r in num_items])
print(f"  Vanilla:     {v_num_acc:.1%} ({sum(r['v_num_ok'] for r in num_items)}/{len(num_items)})")
print(f"  Contrastive: {c_num_acc:.1%} ({sum(r['c_num_ok'] for r in num_items)}/{len(num_items)})")
print(f"  Delta:       {c_num_acc - v_num_acc:+.1%}")

col_items = [r for r in results if r['v_col_ok'] is not None]
if col_items:
    v_col_acc = np.mean([r['v_col_ok'] for r in col_items])
    c_col_acc = np.mean([r['c_col_ok'] for r in col_items])
    print(f"\nCOLOR ACCURACY (samples with colors in reference)")
    print(f"  Vanilla:     {v_col_acc:.1%} ({sum(r['v_col_ok'] for r in col_items)}/{len(col_items)})")
    print(f"  Contrastive: {c_col_acc:.1%} ({sum(r['c_col_ok'] for r in col_items)}/{len(col_items)})")
    print(f"  Delta:       {c_col_acc - v_col_acc:+.1%}")
