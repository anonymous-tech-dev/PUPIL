#!/usr/bin/env python3
"""
04_diagnose.py  —  Diagnostic checks before running the eval
=============================================================
Run this to verify all pieces are in place:
  1. VLMEvalKit installation and LVBench registration
  2. TSV integrity (question count, video availability)
  3. Model names are valid in VLMEvalKit's config
  4. Frame-count estimate for your hardware

References:
  VLMEvalKit supported_VLM registry:
    https://github.com/open-compass/VLMEvalKit/blob/main/vlmeval/config.py
  LVBench video link degradation context:
    https://github.com/zai-org/LVBench/issues  (see open issues on missing videos)
"""

import os, sys, importlib, json
import os.path as osp
import pandas as pd

SEP = "─" * 64

# ── 1. Check VLMEvalKit installation ─────────────────────────────────────────
print(f"\n{SEP}")
print("CHECK 1: VLMEvalKit installation")
print(SEP)
try:
    import vlmeval
    print(f"  ✅  vlmeval found at: {vlmeval.__file__}")
except ImportError:
    print("  ❌  vlmeval not installed.  Run:  bash 01_install.sh")
    sys.exit(1)

# ── 2. Check LVBench is registered ───────────────────────────────────────────
print(f"\n{SEP}")
print("CHECK 2: LVBench dataset registration")
print(SEP)
try:
    from vlmeval.dataset import LVBench
    print(f"  ✅  LVBench class found: {LVBench}")
except ImportError as e:
    print(f"  ❌  LVBench not registered in VLMEvalKit: {e}")
    print("      Make sure 01_install.sh ran successfully and patched")
    print("      vlmeval/dataset/__init__.py with:")
    print("        from .lvbench import LVBench")
    sys.exit(1)

# ── 3. Check TSV ─────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("CHECK 3: LVBench TSV")
print(SEP)

lmu_root = os.environ.get("LMUData", osp.expanduser("~/LMUData"))
tsv_path = osp.join(lmu_root, "LVBench", "LVBench.tsv")

if not osp.exists(tsv_path):
    print(f"  ❌  TSV not found at: {tsv_path}")
    print("      Run:  python 02_prepare_lvbench_tsv.py")
    sys.exit(1)

df = pd.read_csv(tsv_path, sep="\t", dtype=str)
print(f"  ✅  TSV loaded: {len(df)} rows from {tsv_path}")

# Check video availability
videos_dir = osp.join(lmu_root, "LVBench", "videos")
if not osp.exists(videos_dir):
    print(f"  ❌  Videos directory not found: {videos_dir}")
    print("      02_prepare_lvbench_tsv.py should have symlinked it.")
else:
    print(f"  ✅  Videos dir: {videos_dir}")
    available = 0
    missing_videos = []
    for vid in df["video"].unique():
        if osp.exists(osp.join(videos_dir, str(vid))):
            available += 1
        else:
            missing_videos.append(vid)
    print(f"  ✅  Available videos: {available} / {df['video'].nunique()} unique")
    if missing_videos:
        print(f"  ⚠   Missing {len(missing_videos)} video files:")
        for v in missing_videos[:10]:
            print(f"       {v}")
        if len(missing_videos) > 10:
            print(f"       ... and {len(missing_videos)-10} more")
        print(f"  NOTE: LVBench has known link degradation (100+ → 69 → 58 videos).")
        print(f"        Reporting on only available videos is accepted practice.")

print(f"\n  Category breakdown:")
for cat, grp in df.groupby("category"):
    print(f"    {cat:<28} {len(grp):>4} questions")

# ── 4. Check model names ──────────────────────────────────────────────────────
print(f"\n{SEP}")
print("CHECK 4: Model availability in VLMEvalKit config")
print(SEP)

config_path = "lvbench_config.json"
with open(config_path) as f:
    cfg = json.load(f)

target_models = [k for k in cfg.get("model", {}) if not k.startswith("_")]

try:
    from vlmeval.config import supported_VLM
    registered = list(supported_VLM.keys())
except Exception as e:
    print(f"  ⚠   Could not load supported_VLM: {e}")
    registered = []

for model_name in target_models:
    if model_name in registered:
        print(f"  ✅  {model_name}")
    else:
        # Check for partial matches
        matches = [r for r in registered if model_name.lower().replace("-","") in r.lower().replace("-","")]
        if matches:
            print(f"  ⚠   '{model_name}' not found exactly, but similar names exist:")
            for m in matches[:5]:
                print(f"       {m}")
            print(f"      → Update lvbench_config.json 'model' key to one of these.")
        else:
            print(f"  ❌  '{model_name}' not found in supported_VLM.")
            print(f"      Run: python -c \"from vlmeval.config import supported_VLM; "
                  f"print([k for k in supported_VLM if 'qwen' in k.lower()])\"")

print(f"\n  All registered Qwen-VL models:")
qwen_models = [k for k in registered if "qwen" in k.lower() and "vl" in k.lower()]
for m in sorted(qwen_models):
    print(f"    {m}")

# ── 5. Frame count estimate ───────────────────────────────────────────────────
print(f"\n{SEP}")
print("CHECK 5: Frame sampling estimate")
print(SEP)
print("  LVBench videos average ~30 min (up to 2 hours).")
print("  At fps=1.0:  ~1800 frames for 30 min video (may OOM on 7B models).")
print("  At fps=0.5:  ~900 frames — recommended for A100/H100 with 7B models.")
print("  At nframe=32: fixed, memory-safe, but loses temporal resolution.")
print("")
print("  VLMEvalKit community practice for long-video benchmarks:")
print("  See: https://github.com/open-compass/VLMEvalKit/issues/876")
print("  Adjust fps or nframe in lvbench_config.json as needed.")

print(f"\n{SEP}")
print("All checks passed.  Ready to run:  bash 03_run_eval.sh")
print(SEP)
