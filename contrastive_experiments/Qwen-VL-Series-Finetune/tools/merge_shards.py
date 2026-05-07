"""
Merge sharded predictions & metrics from a distributed eval run.

Given an output directory containing:
    predictions.shard0ofN.json, predictions.shard1ofN.json, ...
    metrics.shard0ofN.json,     metrics.shard1ofN.json, ...

Produces:
    predictions.json  — concatenated sample_results (in original test order)
    metrics.json      — re-aggregated metrics over the merged predictions

After a successful merge, the shard files are deleted.

Usage:
    python tools/merge_shards.py --output_dir /path/to/test_results --num_shards 4
"""
import argparse
import json
import os
from collections import defaultdict


def aggregate(all_sample_metrics, metadata_list):
    if not all_sample_metrics:
        return {}
    keys = [k for k in all_sample_metrics[0] if isinstance(all_sample_metrics[0][k], (int, float))]
    if not keys:
        return {}

    overall = {}
    for k in keys:
        vals = [m[k] for m in all_sample_metrics if k in m]
        overall[k] = sum(vals) / len(vals) if vals else 0.0

    per_source = defaultdict(lambda: defaultdict(list))
    for m, meta in zip(all_sample_metrics, metadata_list):
        src = (meta or {}).get("source", "unknown")
        for k in keys:
            if k in m:
                per_source[src][k].append(m[k])

    per_source_agg = {}
    for src, kv in per_source.items():
        per_source_agg[src] = {k: sum(v) / len(v) for k, v in kv.items()}
        per_source_agg[src]["count"] = len(next(iter(kv.values())))

    return {"overall": overall, "per_source": per_source_agg, "total": len(all_sample_metrics)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--num_shards", type=int, required=True)
    args = ap.parse_args()

    # ── Load & concatenate predictions (shard 0, 1, ... N-1 → original order) ──
    merged_predictions = []
    shard_metrics = []
    missing = []
    for i in range(args.num_shards):
        pred_path = os.path.join(args.output_dir, f"predictions.shard{i}of{args.num_shards}.json")
        met_path = os.path.join(args.output_dir, f"metrics.shard{i}of{args.num_shards}.json")
        if not os.path.exists(pred_path):
            missing.append(pred_path)
            continue
        with open(pred_path) as f:
            merged_predictions.extend(json.load(f))
        if os.path.exists(met_path):
            with open(met_path) as f:
                shard_metrics.append(json.load(f))

    if missing:
        print(f"ERROR: {len(missing)} shard prediction file(s) missing:")
        for m in missing:
            print(f"  {m}")
        raise SystemExit(1)

    print(f"Merged {len(merged_predictions)} predictions from {args.num_shards} shards")

    # ── Re-aggregate metrics over all merged predictions ──
    all_sample_metrics = [p.get("metrics", {}) for p in merged_predictions]
    metadata_list = [p.get("metadata", {}) for p in merged_predictions]
    merged_metrics = aggregate(all_sample_metrics, metadata_list)

    # Preserve some top-level fields from shard 0 (model_id, adapter_path)
    if shard_metrics:
        for k in ("model_id", "adapter_path"):
            if k in shard_metrics[0]:
                merged_metrics[k] = shard_metrics[0][k]
        # Sum wall-clock time across shards (wall time is max, sample-seconds is sum)
        total_elapsed = max((m.get("elapsed_seconds", 0) for m in shard_metrics), default=0)
        merged_metrics["elapsed_seconds"] = total_elapsed
        merged_metrics["samples_per_second"] = (
            len(merged_predictions) / total_elapsed if total_elapsed > 0 else 0
        )
        merged_metrics["num_shards"] = args.num_shards

    # ── Write merged outputs ──
    out_pred = os.path.join(args.output_dir, "predictions.json")
    out_met = os.path.join(args.output_dir, "metrics.json")
    with open(out_pred, "w") as f:
        json.dump(merged_predictions, f, indent=2, ensure_ascii=False)
    with open(out_met, "w") as f:
        json.dump(merged_metrics, f, indent=2)
    print(f"Wrote merged: {out_pred}")
    print(f"Wrote merged: {out_met}")

    # ── Delete shard files ──
    for i in range(args.num_shards):
        for base in ("predictions", "metrics"):
            shard_path = os.path.join(args.output_dir, f"{base}.shard{i}of{args.num_shards}.json")
            if os.path.exists(shard_path):
                os.remove(shard_path)
    print(f"Deleted {args.num_shards * 2} shard files.")

    # ── Print summary ──
    overall = merged_metrics.get("overall", {})
    if overall:
        print("\n" + "=" * 60)
        print("MERGED EVALUATION RESULTS")
        print("=" * 60)
        for k, v in sorted(overall.items()):
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
