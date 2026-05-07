"""
run_eval.py — Run VLMEvalKit evaluation on LVBench v2 with local models.

This script monkey-patches VLMEvalKit's dataset registry to include our
LVBench v2 dataset, then delegates to VLMEvalKit's standard run.py pipeline
so that model loading, prompting, inference, and evaluation all follow the
exact same code paths used by the community.

Usage (after setup.sh):
    # Qwen2.5-VL-7B baseline
    python run_eval.py --model Qwen2.5-VL-7B-Instruct --fps 1.0

    # Qwen3-VL-8B baseline
    python run_eval.py --model Qwen3-VL-8B-Instruct --fps 1.0
"""

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def patch_registry():
    """Register LVBench_v2_MCQ in VLMEvalKit's dataset + video_dataset lookups."""
    from functools import partial

    # 1. Import our custom class
    sys.path.insert(0, SCRIPT_DIR)
    from lvbench_v2_dataset import LVBenchV2

    # 2. Patch SUPPORTED_DATASETS (used by build_dataset)
    import vlmeval.dataset as ds_module
    ds_module.LVBenchV2 = LVBenchV2
    if "LVBench_v2_MCQ" not in ds_module.SUPPORTED_DATASETS:
        ds_module.SUPPORTED_DATASETS["LVBench_v2_MCQ"] = "LVBenchV2"

    # 3. Patch the video dataset config so run.py picks it up
    from vlmeval.dataset import video_dataset_config as vdc
    vdc.supported_video_datasets["LVBench_v2_MCQ_1fps"] = partial(
        LVBenchV2, dataset="LVBench_v2_MCQ", fps=1.0
    )
    vdc.supported_video_datasets["LVBench_v2_MCQ_2fps"] = partial(
        LVBenchV2, dataset="LVBench_v2_MCQ", fps=2.0
    )
    vdc.supported_video_datasets["LVBench_v2_MCQ_64frame"] = partial(
        LVBenchV2, dataset="LVBench_v2_MCQ", nframe=64
    )
    vdc.supported_video_datasets["LVBench_v2_MCQ_128frame"] = partial(
        LVBenchV2, dataset="LVBench_v2_MCQ", nframe=128
    )

    # 4. Also patch into the VIDEO_DATASET list (used for build_dataset)
    ds_module.VIDEO_DATASET.append(LVBenchV2)

    print("[patch] Registered LVBench_v2_MCQ variants in VLMEvalKit.")


def main():
    parser = argparse.ArgumentParser(
        description="Run VLMEvalKit evaluation on LVBench v2"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="VLMEvalKit model name, e.g. Qwen2.5-VL-7B-Instruct or Qwen3-VL-8B-Instruct"
    )
    parser.add_argument(
        "--fps", type=float, default=1.0,
        help="Frames per second for video sampling (default: 1.0)"
    )
    parser.add_argument(
        "--nframe", type=int, default=0,
        help="Fixed number of frames (overrides --fps if > 0)"
    )
    parser.add_argument(
        "--work-dir", type=str, default=os.path.join(SCRIPT_DIR, "outputs"),
        help="Output directory for predictions and scores"
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Determine dataset variant name
    if args.nframe > 0:
        dataset_name = f"LVBench_v2_MCQ_{args.nframe}frame"
    else:
        fps_tag = str(args.fps).replace(".", "")
        dataset_name = f"LVBench_v2_MCQ_{fps_tag}fps"
        # Ensure the fps variant is registered
        # (patch_registry only registers 1.0 and 2.0 by default)

    # Patch the registry before importing run logic
    patch_registry()

    # Also register the exact variant if not already
    from functools import partial
    from vlmeval.dataset.video_dataset_config import supported_video_datasets
    sys.path.insert(0, SCRIPT_DIR)
    from lvbench_v2_dataset import LVBenchV2

    if dataset_name not in supported_video_datasets:
        if args.nframe > 0:
            supported_video_datasets[dataset_name] = partial(
                LVBenchV2, dataset="LVBench_v2_MCQ", nframe=args.nframe
            )
        else:
            supported_video_datasets[dataset_name] = partial(
                LVBenchV2, dataset="LVBench_v2_MCQ", fps=args.fps
            )

    # Construct the VLMEvalKit command-line args
    sys.argv = [
        "run.py",
        "--data", dataset_name,
        "--model", args.model,
        "--work-dir", args.work_dir,
        "--mode", "all",
    ]
    if args.verbose:
        sys.argv.append("--verbose")

    print(f"\n{'='*65}")
    print(f"  VLMEvalKit — LVBench v2 Evaluation")
    print(f"  Model   : {args.model}")
    print(f"  Dataset : {dataset_name}")
    print(f"  Work dir: {args.work_dir}")
    print(f"{'='*65}\n")

    # Import and call VLMEvalKit's main
    # We need to find run.py in the VLMEvalKit package
    import vlmeval
    vlmeval_root = os.path.dirname(os.path.dirname(vlmeval.__file__))
    run_py = os.path.join(vlmeval_root, "run.py")

    if os.path.exists(run_py):
        # Run VLMEvalKit's run.py main()
        import importlib.util
        spec = importlib.util.spec_from_file_location("vlmeval_run", run_py)
        run_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run_module)
        run_module.main()
    else:
        # Fallback: use vlmeval.tools CLI
        from vlmeval.smp import load_env
        load_env()
        # Direct import
        from vlmeval.config import supported_VLM
        from vlmeval.dataset import build_dataset
        from vlmeval.inference_video import infer_data_job_video

        print(f"[run_eval] Building dataset: {dataset_name}")
        dataset = supported_video_datasets[dataset_name]()

        print(f"[run_eval] Loading model: {args.model}")
        assert args.model in supported_VLM, (
            f"Model '{args.model}' not in supported_VLM. "
            f"Available Qwen models: {[k for k in supported_VLM if 'qwen' in k.lower() or 'Qwen' in k]}"
        )
        model_builder = supported_VLM[args.model]
        model = model_builder()

        os.makedirs(args.work_dir, exist_ok=True)
        result_file = os.path.join(
            args.work_dir, f"{args.model}_{dataset_name}.xlsx"
        )

        model = infer_data_job_video(
            model,
            work_dir=args.work_dir,
            model_name=args.model,
            dataset=dataset,
            result_file_name=f"{args.model}_{dataset_name}.xlsx",
            verbose=args.verbose,
        )

        # Evaluate
        print(f"\n[run_eval] Evaluating...")
        eval_result = LVBenchV2.evaluate(result_file)
        print(f"[run_eval] Done. Results at: {args.work_dir}")


if __name__ == "__main__":
    main()
