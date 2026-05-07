import json, os
from collections import Counter
import numpy as np

tcot_f = '/workspace/Pupil/frame_sampling_experiments/temporal_cot_gdm/results/lvbench_v2/Qwen3-VL-8B_dynamic_segment_l12_s64_k256_u32_results_v6.jsonl'
ours_f = 'results/lvbench_v2/Qwen3-VL-8B_seg_scene_detect_nosel_k768_u96_results.jsonl'

def load(f):
    store = {}
    with open(f) as fh:
        for line in fh:
            if not line.strip(): continue
            d = json.loads(line)
            store[str(d['uid'])] = d
    return store

tcot = load(tcot_f)
ours = load(ours_f)
common = set(tcot) & set(ours)

buckets = {'tcot_only':[], 'ours_only':[], 'both_wrong':[], 'both_right':[]}
for uid in common:
    t_ok = tcot[uid].get('predicted_letter','') == tcot[uid].get('ground_truth','')
    o_ok = ours[uid].get('predicted_letter','') == ours[uid].get('ground_truth','')
    if t_ok and o_ok: buckets['both_right'].append(uid)
    elif t_ok and not o_ok: buckets['tcot_only'].append(uid)
    elif not t_ok and o_ok: buckets['ours_only'].append(uid)
    else: buckets['both_wrong'].append(uid)

print(f'Both right: {len(buckets["both_right"])} | Ours only: {len(buckets["ours_only"])} | TCoT only: {len(buckets["tcot_only"])} | Both wrong: {len(buckets["both_wrong"])}')

# TOP FAILING VIDEOS
print('\n=== TOP VIDEOS WHERE TCOT BEATS US (68 questions) ===')
vid_fails = Counter()
for uid in buckets['tcot_only']:
    vid = os.path.basename(ours[uid]['video_path'])
    vid_fails[vid] += 1
for vid, cnt in vid_fails.most_common(10):
    tf = ours[[u for u in buckets['tcot_only'] if os.path.basename(ours[u]['video_path'])==vid][0]]['total_frames']
    print(f'  {vid[:50]:50s} fails={cnt} total_frames={tf}')

# Coverage ratio
print('\n=== Coverage ratio (ctx/total_frames) ===')
for cat in ['tcot_only','ours_only','both_wrong','both_right']:
    ratios = []
    for uid in buckets[cat]:
        ctx = ours[uid].get('num_context',0)
        tf = ours[uid].get('total_frames',0)
        if tf > 0: ratios.append(ctx/tf)
    if ratios:
        print(f'  {cat:15s}: {np.mean(ratios)*100:.2f}% coverage (n={len(ratios)})')

# TCoT context for tcot-only wins
print('\n=== TCoT vs Ours context frames for tcot-only wins ===')
tcot_ctx = [tcot[uid].get('num_context',0) for uid in buckets['tcot_only']]
ours_ctx = [ours[uid].get('num_context',0) for uid in buckets['tcot_only']]
print(f'  TCoT ctx: mean={np.mean(tcot_ctx):.0f}')
print(f'  Ours ctx: mean={np.mean(ours_ctx):.0f}')

# Question type breakdown
print('\n=== TCOT-ONLY wins by question type ===')
tc = Counter()
for uid in buckets['tcot_only']:
    qtypes = ours[uid].get('question_type', [])
    if isinstance(qtypes, str): qtypes = [qtypes]
    for qt in qtypes: tc[qt] += 1
for qt, c in tc.most_common():
    print(f'  {qt}: {c}')

# Entity recognition deep dive
print('\n=== ENTITY RECOGNITION deep dive ===')
er_all = [uid for uid in common if 'entity recognition' in (ours[uid].get('question_type',[]) if isinstance(ours[uid].get('question_type',[]), list) else [ours[uid].get('question_type','')])]
er_ours = sum(1 for uid in er_all if ours[uid]['predicted_letter']==ours[uid]['ground_truth'])
er_tcot = sum(1 for uid in er_all if tcot[uid]['predicted_letter']==tcot[uid]['ground_truth'])
print(f'  Total: {len(er_all)} | Ours: {er_ours} ({100*er_ours/len(er_all):.1f}%) | TCoT: {er_tcot} ({100*er_tcot/len(er_all):.1f}%)')

# BOTH-WRONG analysis
print('\n=== BOTH-WRONG by question type (285 questions) ===')
type_counts = Counter()
total_by_type = Counter()
for uid in common:
    qtypes = ours[uid].get('question_type', [])
    if isinstance(qtypes, str): qtypes = [qtypes]
    for qt in qtypes:
        total_by_type[qt] += 1
for uid in buckets['both_wrong']:
    qtypes = ours[uid].get('question_type', [])
    if isinstance(qtypes, str): qtypes = [qtypes]
    for qt in qtypes:
        type_counts[qt] += 1
for qt, c in type_counts.most_common():
    print(f'  {qt:30s}: {c:3d}/{total_by_type[qt]:3d} = {100*c/total_by_type[qt]:.1f}% both-fail')

# Answer bias
print('\n=== ANSWER BIAS (tcot-only failures) ===')
our_preds = Counter(ours[uid]['predicted_letter'] for uid in buckets['tcot_only'])
gt_dist = Counter(ours[uid]['ground_truth'] for uid in buckets['tcot_only'])
print(f'  Our predictions: {dict(sorted(our_preds.items()))}')
print(f'  Ground truth:    {dict(sorted(gt_dist.items()))}')

print('\n=== OVERALL ANSWER DISTRIBUTION ===')
our_all = Counter(ours[uid]['predicted_letter'] for uid in common)
gt_all = Counter(ours[uid]['ground_truth'] for uid in common)
print(f'  Our preds: {dict(sorted(our_all.items()))}')
print(f'  GT:        {dict(sorted(gt_all.items()))}')

# Are tcot-only wins clustered?
print(f'\n=== CLUSTERING ===')
print(f'  68 tcot-only wins across {len(vid_fails)} unique videos')
print(f'  Top 3 videos account for {sum(c for _,c in vid_fails.most_common(3))}/68 failures')

# Key insight: look at sample questions where TCoT wins
print('\n=== SAMPLE QUESTIONS WHERE TCOT WINS ===')
for uid in buckets['tcot_only'][:12]:
    d = ours[uid]
    qt = d.get('question_type', [])
    print(f'  [{",".join(qt) if isinstance(qt,list) else qt}] Q: {d["question"][:100]}')
    print(f'    GT={d["ground_truth"]} Ours={d["predicted_letter"]} TCoT={tcot[uid]["predicted_letter"]} ctx={d["num_context"]}')
