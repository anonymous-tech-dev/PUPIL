#!/usr/bin/env bash
# ==============================================================================
#  training_watchdog.sh
#  Live health monitor for SFT / DPO runs.
#
#  Watches:
#    - host RAM (kills run if > RAM_KILL_GB; warns at RAM_WARN_GB)
#    - per-GPU memory + utilization
#    - training.log loss / grad_norm (alerts on NaN/Inf, loss spike, torchvision fallback)
#
#  Usage:
#    bash training_watchdog.sh <output_dir> [interval_sec]
#
#  Examples:
#    bash training_watchdog.sh \
#      /workspace/Pupil/contrastive_experiments/outputs/sof_sft_warmstart_lr5e-6_ep3_bs64_24576seq_16fr
#    bash training_watchdog.sh "$OUTPUT_DIR" 5
#
#  Env knobs:
#    RAM_WARN_GB   default 1500   warn at this host RAM use
#    RAM_KILL_GB   default 1850   auto-kill training above this (pod has 2T)
#    LOSS_SPIKE    default 5.0    alert if any loss above this value
#    AUTO_KILL     default 0      set 1 to actually pkill on RAM_KILL_GB
# ==============================================================================
set -u

OUT_DIR="${1:?Usage: bash training_watchdog.sh <output_dir> [interval_sec]}"
INTERVAL="${2:-10}"

RAM_WARN_GB="${RAM_WARN_GB:-1500}"
RAM_KILL_GB="${RAM_KILL_GB:-1850}"
LOSS_SPIKE="${LOSS_SPIKE:-5.0}"
AUTO_KILL="${AUTO_KILL:-0}"

LOG_FILE="$OUT_DIR/training.log"

CR='\033[0;31m'   # red
CY='\033[0;33m'   # yellow
CG='\033[0;32m'   # green
CB='\033[0;36m'   # cyan
CW='\033[1;37m'   # white
NC='\033[0m'      # reset

ts() { date +'%H:%M:%S'; }

print_header() {
    clear
    echo -e "${CW}═══════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${CW}  Training Watchdog — $(ts)${NC}"
    echo -e "${CW}  OUT: ${CB}$OUT_DIR${NC}"
    echo -e "${CW}  RAM warn>${RAM_WARN_GB}G  kill>${RAM_KILL_GB}G  loss-spike>${LOSS_SPIKE}  auto_kill=${AUTO_KILL}${NC}"
    echo -e "${CW}═══════════════════════════════════════════════════════════════════════${NC}"
}

check_ram() {
    local used_g free_g total_g pct
    read used_g free_g total_g <<<"$(free -g | awk '/^Mem:/ {print $3, $4, $2}')"
    pct=$(( used_g * 100 / total_g ))
    local color="$CG"; local tag="OK"
    if   (( used_g > RAM_KILL_GB )); then color="$CR"; tag="KILL"
    elif (( used_g > RAM_WARN_GB )); then color="$CY"; tag="WARN"
    fi
    echo -e "  ${CW}RAM${NC}   ${color}${used_g}G / ${total_g}G  (${pct}%)  [$tag]${NC}"

    if (( used_g > RAM_KILL_GB )) && [[ "$AUTO_KILL" == "1" ]]; then
        echo -e "${CR}  🛑 RAM > ${RAM_KILL_GB}G — auto-killing training!${NC}"
        pkill -9 -f "deepspeed\|torch\.distributed\|train_sft\|sof_dpo_train" 2>/dev/null || true
    fi
}

check_gpu() {
    if ! command -v nvidia-smi >/dev/null; then return; fi
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
               --format=csv,noheader,nounits 2>/dev/null | \
    while IFS=',' read -r idx used total util; do
        used=$(echo "$used" | tr -d ' ')
        total=$(echo "$total" | tr -d ' ')
        util=$(echo "$util" | tr -d ' ')
        local pct=0
        [[ "$total" -gt 0 ]] && pct=$(( used * 100 / total ))
        local color="$CG"
        (( pct > 95 )) && color="$CR"
        (( pct > 85 && pct <= 95 )) && color="$CY"
        printf "  ${CW}GPU${NC} %s  ${color}%5sM / %5sM (%2s%%)${NC}  util=%3s%%\n" \
            "$idx" "$used" "$total" "$pct" "$util"
    done
}

check_log() {
    if [[ ! -f "$LOG_FILE" ]]; then
        echo -e "  ${CY}log not yet created: $LOG_FILE${NC}"
        return
    fi

    # Most recent loss line
    local last_loss_line
    last_loss_line=$(grep -E "'loss':" "$LOG_FILE" 2>/dev/null | tail -1)
    if [[ -n "$last_loss_line" ]]; then
        local loss step gn
        loss=$(echo "$last_loss_line" | grep -oE "'loss': [-0-9.]+" | awk '{print $2}')
        step=$(echo "$last_loss_line" | grep -oE "'step': [0-9]+"   | awk '{print $2}')
        gn=$(  echo "$last_loss_line" | grep -oE "'grad_norm': [-0-9.eE+]+" | awk '{print $2}')

        local color="$CG"; local tag="OK"
        if [[ "$loss" == *nan* || "$loss" == *NaN* || "$loss" == *inf* ]]; then
            color="$CR"; tag="NaN/Inf 💀"
        elif awk "BEGIN{exit !($loss > $LOSS_SPIKE)}"; then
            color="$CR"; tag="SPIKE 💥"
        elif awk "BEGIN{exit !($loss < 0.001)}"; then
            color="$CR"; tag="DEAD (loss≈0) 💀"
        fi
        echo -e "  ${CW}TRAIN${NC} step=${step:-?}  loss=${color}${loss}${NC}  grad_norm=${gn:-?}  [$tag]"
    else
        echo -e "  ${CW}TRAIN${NC} ${CY}no loss lines yet${NC}"
    fi

    # Most recent eval_loss
    local last_eval
    last_eval=$(grep -E "'eval_loss':" "$LOG_FILE" 2>/dev/null | tail -1)
    if [[ -n "$last_eval" ]]; then
        local el estep
        el=$(   echo "$last_eval" | grep -oE "'eval_loss': [-0-9.a-zA-Z]+" | awk '{print $2}')
        estep=$(echo "$last_eval" | grep -oE "'epoch': [0-9.]+"            | awk '{print $2}')
        local color="$CG"
        [[ "$el" == *nan* || "$el" == *NaN* || "$el" == *inf* ]] && color="$CR"
        echo -e "  ${CW}EVAL${NC}  epoch=${estep:-?}  eval_loss=${color}${el}${NC}"
    fi

    # Bad keywords in last 50 lines
    local bad
    bad=$(tail -50 "$LOG_FILE" | grep -ciE "torchvision|fallback|cuda out of memory|killed|RuntimeError|nan loss|inf loss" || true)
    if [[ "$bad" -gt 0 ]]; then
        echo -e "  ${CR}⚠️  $bad suspicious lines in last 50 (torchvision/OOM/RuntimeError)${NC}"
        tail -50 "$LOG_FILE" | grep -iE "torchvision|fallback|cuda out of memory|killed|RuntimeError|nan loss|inf loss" | head -3 | sed 's/^/      /'
    fi

    # Latest checkpoint
    local latest_ckpt
    latest_ckpt=$(ls -1d "$OUT_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -1)
    if [[ -n "$latest_ckpt" ]]; then
        local ckpt_age
        ckpt_age=$(( $(date +%s) - $(stat -c %Y "$latest_ckpt" 2>/dev/null || echo 0) ))
        echo -e "  ${CW}CKPT${NC}  $(basename "$latest_ckpt")  (${ckpt_age}s ago)"
    fi
}

check_procs() {
    local n
    n=$(pgrep -af "deepspeed\|train_sft\|sof_dpo_train" 2>/dev/null | wc -l)
    if [[ "$n" -eq 0 ]]; then
        echo -e "  ${CY}⚠️  no training process detected${NC}"
    else
        echo -e "  ${CW}PROC${NC}  ${CG}$n${NC} training process(es) alive"
    fi
}

# ── main loop ───────────────────────────────────────────────────────────
trap 'echo; echo "watchdog stopped"; exit 0' INT TERM

while true; do
    print_header
    check_ram
    echo
    check_gpu
    echo
    check_procs
    echo
    check_log
    echo
    echo -e "  ${CW}refresh in ${INTERVAL}s — Ctrl+C to stop${NC}"
    sleep "$INTERVAL"
done
