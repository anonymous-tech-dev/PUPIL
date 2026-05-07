#!/bin/bash
# Wait for GPU 3 dhash_nosel_k256 to finish, then launch uniform_nosel_k512 (no kf)
echo "[$(date)] Waiting for GPU 3 experiment to finish..."

while ps aux | grep "python main.py" | grep -v grep | grep "cuda_visible_devices=3" > /dev/null 2>&1; do
    sleep 60
done

echo "[$(date)] GPU 3 free. Launching uniform + nosel + k512 (no kf filter)..."
cd /workspace/Pupil/frame_sampling_experiments/edu_cot
nohup python main.py \
    pipeline.segmentation=uniform \
    pipeline.keyframe_filter=false \
    pipeline.vlm_selection=false \
    aggregation.context_budget_frames=512 \
    aggregation.uniform_context_frames=64 \
    cuda_visible_devices=3 \
    > /tmp/overnight_uniform_nofilt_k512_gpu3.log 2>&1 &

echo "[$(date)] GPU 3 launched PID=$!"
