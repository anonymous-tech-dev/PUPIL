#!/bin/bash
# v2_run_all.sh — End-to-end orchestrator for the v2 SoF-DPO data pipeline.
#
# Stages
#   00  generate negatives  (8-GPU data parallel; ~5-6h)
#   01  ROUGE/keyword/abstain filter   (CPU; <1min)
#   02  reference-policy margins (8-GPU data parallel; ~2-3h)
#   03  assemble pre-judge DPO + SFT   (CPU; <1min)
#   04  GPT-5 judge             (Azure; ~10-30min)
#   05  apply judge → judged DPO + SFT (CPU; <1s)
#   06  curriculum re-order      (CPU; <1s)  -> Run-2 dataset
#
# Use STAGES=0,1,2 to run a subset; default = all.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPTS="$REPO/scripts"
STAGES="${STAGES:-0,1,2,3,4,5,6}"

run_stage() {
    local n="$1"
    case "$n" in
      0) bash "$SCRIPTS/v2_00_build_negatives.sh" ;;
      1) bash "$SCRIPTS/v2_01_filter.sh" ;;
      2) bash "$SCRIPTS/v2_02_score_margins.sh" ;;
      3) bash "$SCRIPTS/v2_03_assemble.sh" ;;
      4) bash "$SCRIPTS/v2_04_judge.sh" ;;
      5) bash "$SCRIPTS/v2_05_apply_judge.sh" ;;
      6) bash "$SCRIPTS/v2_06_curriculum.sh" ;;
      *) echo "unknown stage: $n"; exit 2 ;;
    esac
}

IFS=',' read -r -a STAGE_ARR <<< "$STAGES"
for s in "${STAGE_ARR[@]}"; do
    s="${s// /}"
    [[ -z "$s" ]] && continue
    echo
    echo "════════════════════════════════════════════════════════════════════"
    echo "  v2 STAGE $s   $(date)"
    echo "════════════════════════════════════════════════════════════════════"
    t0=$(date +%s)
    if ! run_stage "$s"; then
        echo "STAGE $s FAILED — aborting."
        exit 1
    fi
    dt=$(( $(date +%s) - t0 ))
    echo "  → stage $s OK in ${dt}s"
done

echo
echo "════════════════════════════════════════════════════════════════════"
echo "  v2 PIPELINE COMPLETE"
echo "════════════════════════════════════════════════════════════════════"
echo "  Run-1 (random shuffle) :"
echo "    DPO  : $REPO/old_dpo_revised_data_8b/sof_dpo_train.judged.json"
echo "    SFT  : $REPO/old_dpo_revised_data_8b/sof_sft_warmstart.no_transcript.judged.json"
echo "  Run-2 (easy→hard curriculum) :"
echo "    DPO  : $REPO/old_dpo_revised_data_8b/sof_dpo_train.judged.curriculum.json"
echo "    SFT  : $REPO/old_dpo_revised_data_8b/sof_sft_warmstart.no_transcript.judged.curriculum.json"
