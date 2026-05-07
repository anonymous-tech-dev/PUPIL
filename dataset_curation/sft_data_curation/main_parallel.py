"""
main_parallel.py -- Per-GPU data-parallel launcher for main.py
================================================================
Run from this directory with the same knobs/env vars as main.py:

    CUDA_VISIBLE_DEVICES="0,1,2,3" python main_parallel.py

This script splits the active [START_IDX, END_IDX) range evenly across the
visible GPUs and starts one independent main.py process per GPU. Each worker
sees exactly one GPU, so each process loads its own model replica on cuda:0.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import main as config


def _parse_visible_gpus(value: str) -> list[str]:
    return [gpu.strip() for gpu in value.split(",") if gpu.strip()]


def _split_range(start: int, end: int, parts: int) -> list[tuple[int, int]]:
    total = end - start
    base = total // parts
    remainder = total % parts

    shards: list[tuple[int, int]] = []
    cursor = start
    for worker_idx in range(parts):
        shard_size = base + (1 if worker_idx < remainder else 0)
        shard_start = cursor
        shard_end = cursor + shard_size
        if shard_start < shard_end:
            shards.append((shard_start, shard_end))
        cursor = shard_end
    return shards


def _dataset_size(path: str) -> int:
    with open(path, "r") as f:
        return len(json.load(f))


def main() -> int:
    visible_gpus = _parse_visible_gpus(config.CUDA_VISIBLE_DEVICES)
    if not visible_gpus:
        print("[error] CUDA_VISIBLE_DEVICES is empty; main_parallel.py needs at least one GPU.")
        return 1

    missing = [p for p in [config.CGBENCH_JSON, config.CLUE_VID_DIR, config.SUBTITLE_DIR]
               if not os.path.exists(p)]
    if missing:
        print("[error] The following paths do not exist:")
        for path in missing:
            print(f"  {path}")
        return 1

    total = _dataset_size(config.CGBENCH_JSON)
    start_idx = config.START_IDX
    end_idx = config.END_IDX if config.END_IDX is not None else total
    if start_idx < 0 or end_idx < start_idx or end_idx > total:
        print(f"[error] Invalid index range [{start_idx}, {end_idx}) for dataset size {total}.")
        return 1

    shards = _split_range(start_idx, end_idx, len(visible_gpus))
    if not shards:
        print(f"[error] Empty index range [{start_idx}, {end_idx}). Nothing to launch.")
        return 1

    run_ts = time.strftime("%Y%m%d_%H%M%S")
    log_dir = Path(config.OUTPUT_DIR) / "parallel_logs" / run_ts
    log_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  CGBench -> SFT Data Generator (data parallel)")
    print("=" * 72)
    print(f"  GPUs       : {','.join(visible_gpus)}")
    print(f"  Workers    : {len(shards)}")
    print(f"  Model      : {config.MODEL_KEY}")
    print(f"  Strategy   : {config.STRATEGY}")
    print(f"  Range      : [{start_idx}, {end_idx}) / {total}")
    print(f"  Log dir    : {log_dir}")
    print("=" * 72)

    script_path = Path(__file__).with_name("main.py")
    processes: list[tuple[str, int, int, Path, subprocess.Popen]] = []

    try:
        for worker_idx, (gpu_id, (shard_start, shard_end)) in enumerate(zip(visible_gpus, shards)):
            env = os.environ.copy()
            env.update({
                "CUDA_VISIBLE_DEVICES": gpu_id,
                "START_IDX": str(shard_start),
                "END_IDX": str(shard_end),
                "PYTHONUNBUFFERED": "1",
            })
            log_path = log_dir / f"worker{worker_idx}_gpu{gpu_id}_idx{shard_start}-{shard_end}.log"
            log_file = open(log_path, "w")
            process = subprocess.Popen(
                [sys.executable, str(script_path)],
                cwd=str(Path(__file__).parent),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
            log_file.close()
            processes.append((gpu_id, shard_start, shard_end, log_path, process))
            print(
                f"[launch] worker={worker_idx} gpu={gpu_id} "
                f"idx=[{shard_start}, {shard_end}) pid={process.pid} log={log_path}"
            )

        failures: list[tuple[str, int, int, int, Path]] = []
        for gpu_id, shard_start, shard_end, log_path, process in processes:
            return_code = process.wait()
            if return_code == 0:
                print(f"[done] gpu={gpu_id} idx=[{shard_start}, {shard_end})")
            else:
                failures.append((gpu_id, shard_start, shard_end, return_code, log_path))
                print(
                    f"[fail] gpu={gpu_id} idx=[{shard_start}, {shard_end}) "
                    f"exit={return_code} log={log_path}"
                )

        if failures:
            print("\n[error] One or more workers failed:")
            for gpu_id, shard_start, shard_end, return_code, log_path in failures:
                print(
                    f"  gpu={gpu_id} idx=[{shard_start}, {shard_end}) "
                    f"exit={return_code} log={log_path}"
                )
            return 1

        print("\nAll workers completed successfully.")
        return 0
    except KeyboardInterrupt:
        print("\n[interrupt] Terminating workers...")
        for _, _, _, _, process in processes:
            if process.poll() is None:
                process.terminate()
        for _, _, _, _, process in processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
