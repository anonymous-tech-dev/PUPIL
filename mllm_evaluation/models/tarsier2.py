"""
Tarsier2-7B (omni-research/Tarsier2-7b-0115) evaluator.

Tarsier ships as a repo (cloned at third_party/tarsier/), not a pip package.
We add it to sys.path lazily inside load(), then mirror the official
inference path from third_party/tarsier/tasks/inference_quick_start.py:

  load_model_and_processor(model_path, data_config) -> (model, processor)
  sample = format_one_sample(video_path, prompt)
  batch  = processor(sample)
  out    = model.generate(**batch, **gen_kwargs)
  text   = processor.processor.tokenizer.decode(out[0][len_in:], skip_special_tokens=True)

Greedy decoding, max_new_tokens=512, n_frames=16 (Tarsier paper default).
"""

import os
import sys
import importlib.util
import torch
from models.base import BaseEvaluator


_TARSIER_REPO = "/workspace/Pupil/third_party/tarsier"


def _load_module_from(path: str, name: str):
    """Load a Python file as a uniquely-named module, bypassing the package
    system so we don't collide with `mllm_evaluation/models/`."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _patch_transformers_for_tarsier():
    """Tarsier's processor imports a few helper symbols that were removed
    in transformers >= 4.50.  Stub the missing ones so the import works."""
    import transformers.processing_utils as _pu
    if not hasattr(_pu, "_validate_images_text_input_order"):
        # Identity stub: validation was a deprecation shim in older transformers
        def _validate_images_text_input_order(images, text):
            return images, text
        _pu._validate_images_text_input_order = _validate_images_text_input_order


def _load_tarsier_modules():
    """Manually import the Tarsier source files we need, registering them
    under unique names (tarsier_models, tarsier_dataset, ...) so they don't
    clash with this project's `models/` package."""
    if _TARSIER_REPO not in sys.path:
        sys.path.insert(0, _TARSIER_REPO)
    _patch_transformers_for_tarsier()

    # Tarsier files reference `from models.modeling_tarsier import ...` and
    # `from dataset.utils import ...` internally.  We register aliased
    # top-level packages so those internal imports resolve correctly.
    if "models.modeling_tarsier" in sys.modules:
        return  # already loaded

    # 1. Make the tarsier source dirs importable as top-level packages with
    #    NEW names so they don't clash, BUT also re-register them under the
    #    plain names that tarsier code expects (for the duration of this
    #    process).  We swap sys.modules['models'] / 'dataset' / 'tasks' to
    #    point at the tarsier dirs.
    import types

    def make_pkg(plain_name, dir_path):
        if plain_name in sys.modules:
            saved = sys.modules[plain_name]
        else:
            saved = None
        pkg = types.ModuleType(plain_name)
        pkg.__path__ = [dir_path]
        pkg.__file__ = os.path.join(dir_path, "__init__.py")
        sys.modules[plain_name] = pkg
        return saved

    saved_models  = make_pkg("models",  os.path.join(_TARSIER_REPO, "models"))
    saved_dataset = make_pkg("dataset", os.path.join(_TARSIER_REPO, "dataset"))
    saved_tasks   = make_pkg("tasks",   os.path.join(_TARSIER_REPO, "tasks"))
    saved_tools   = make_pkg("tools",   os.path.join(_TARSIER_REPO, "tools"))

    try:
        # Now ordinary imports inside tarsier code work.
        from models import modeling_tarsier as _mt    # noqa: F401
        from dataset import tarsier_datamodule as _td  # noqa: F401
        from dataset import utils as _du               # noqa: F401
        from tasks import utils as _tu                 # noqa: F401
    finally:
        # Restore our packages so the rest of mllm_evaluation keeps working.
        # We KEEP the just-loaded tarsier submodules in sys.modules under
        # their `models.*` / `dataset.*` keys, but make `sys.modules['models']`
        # point back to OUR package so future `from models.base import ...`
        # works.
        if saved_models  is not None: sys.modules["models"]  = saved_models
        if saved_dataset is not None: sys.modules["dataset"] = saved_dataset
        if saved_tasks   is not None: sys.modules["tasks"]   = saved_tasks
        if saved_tools   is not None: sys.modules["tools"]   = saved_tools


def _patch_tarsier_force_slow_image_processor():
    """Tarsier's data path does:
        if isinstance(self.image_processor, Qwen2VLImageProcessor):
            ... per-frame video branch ...
        else:
            ... single-image branch (broken on Qwen2VLImageProcessorFast) ...

    transformers >= 4.50 makes AutoImageProcessor return *Fast* by default,
    which fails that isinstance check → else branch returns flat patches with
    ndim=1 → get_image_size crashes.

    Fix: monkey-patch AutoImageProcessor.from_pretrained to force
    use_fast=False *only for the qwen2_vl image processor*, while leaving
    the tokenizer as fast (Tarsier needs `char_to_token` which is fast-only).
    """
    # Also remap Tarsier's legacy preprocessor_config.json that ships
    #     "size": {"max_pixels": 2073600, "min_pixels": 3136}
    # which transformers >= 4.50 rejects with
    #     ValueError: size must contain 'shortest_edge' and 'longest_edge' keys.
    # We patch the slow processor's __init__ to remap legacy keys.
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
        Qwen2VLImageProcessor,
    )
    if not getattr(Qwen2VLImageProcessor, "_legacy_size_patched", False):
        _orig_init = Qwen2VLImageProcessor.__init__

        def _patched_init(self, *args, **kwargs):
            sz = kwargs.get("size")
            if isinstance(sz, dict) and (
                "shortest_edge" not in sz or "longest_edge" not in sz
            ):
                # Treat legacy {min_pixels, max_pixels} as pixel-count bounds.
                short = sz.get("min_pixels", sz.get("shortest_edge", 56 * 56))
                long_ = sz.get("max_pixels", sz.get("longest_edge", 28 * 28 * 1280))
                kwargs["size"] = {"shortest_edge": short, "longest_edge": long_}
                # Also forward as top-level kwargs (they win over `size` per
                # the upstream constructor's backward-compat block).
                kwargs.setdefault("min_pixels", short)
                kwargs.setdefault("max_pixels", long_)
            return _orig_init(self, *args, **kwargs)

        Qwen2VLImageProcessor.__init__ = _patched_init
        Qwen2VLImageProcessor._legacy_size_patched = True

    # Force AutoImageProcessor to use the slow Qwen2VL processor.  We do NOT
    # want `use_fast=False` on the AutoTokenizer (it'd return a Python
    # tokenizer that lacks char_to_token, breaking Tarsier's chat template).
    from transformers import AutoImageProcessor as _AutoIP
    if not getattr(_AutoIP.from_pretrained, "_tarsier_patched", False):
        _orig_ip_fp = _AutoIP.from_pretrained.__func__

        @classmethod
        def patched_ip_fp(cls, *args, **kwargs):
            kwargs.setdefault("use_fast", False)
            return _orig_ip_fp(cls, *args, **kwargs)

        patched_ip_fp.__func__._tarsier_patched = True
        _AutoIP.from_pretrained = patched_ip_fp


def _patch_qwen2vl_dynamic_init_weights():
    """Tarsier's Qwen2VLVisionConfig forgets to set `initializer_range`,
    but Qwen2VLPreTrainedModel._init_weights reads `self.config.initializer_range`.
    In transformers >= 4.50 this fires unconditionally during from_pretrained
    via initialize_weights → smart_apply → _initialize_weights.

    Pre-trigger the HF dynamic-module load (cwd must already be tarsier repo
    root), then:
      1. add a class-level default `initializer_range = 0.02` to the vision
         config so existing code paths keep working,
      2. wrap `_init_weights` with a getattr fallback for safety.
    """
    import sys
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    # repo_id="models", class_ref="modeling_qwen2_vl_fast.Qwen2VLVisionConfig"
    # Trigger load (cwd is already tarsier repo via os.chdir in load()).
    get_class_from_dynamic_module(
        "modeling_qwen2_vl_fast.Qwen2VLVisionConfig", "models"
    )
    mod_key = "transformers_modules.models.modeling_qwen2_vl_fast"
    if mod_key not in sys.modules:
        return  # nothing to patch — load will fall over with the original error
    qfast = sys.modules[mod_key]

    # 1. Default initializer_range on the vision config class.
    if hasattr(qfast, "Qwen2VLVisionConfig") and not hasattr(
        qfast.Qwen2VLVisionConfig, "initializer_range"
    ):
        qfast.Qwen2VLVisionConfig.initializer_range = 0.02

    # 2. Defensive _init_weights on the PreTrainedModel base.
    base_cls = getattr(qfast, "Qwen2VLPreTrainedModel", None)
    if base_cls is None or getattr(base_cls, "_init_weights_patched", False):
        return
    import torch.nn as nn
    def _safe_init_weights(self, module):
        std = getattr(self.config, "initializer_range", 0.02)
        if isinstance(module, (nn.Linear, nn.Conv3d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
    base_cls._init_weights = _safe_init_weights
    base_cls._init_weights_patched = True


class Tarsier2Evaluator(BaseEvaluator):
    MODEL_ID = "omni-research/Tarsier2-7b-0115"
    NUM_FRAMES = 16
    MAX_NEW_TOKENS = 512
    CONFIG_PATH = os.path.join(_TARSIER_REPO, "configs", "tarser2_default_config.yaml")

    def load(self):
        import yaml
        _load_tarsier_modules()
        # tasks.utils sticks around as a sys.modules entry after the loader,
        # but only its symbols (re-imported below) are stable.
        from tasks.utils import load_model_and_processor  # type: ignore

        # ── transformers >= 4.57 compatibility patch ─────────────────────
        # TarsierPreTrainedModel defines `_no_split_modules` as a @property
        # that references self.language_model / self.vision_tower.  In modern
        # transformers, PreTrainedModel.__init__ does
        #     self._no_split_modules = self._no_split_modules or []
        # which triggers the property *before* TarsierForConditionalGeneration
        # populates those submodules → AttributeError.  Replace with a plain
        # class attribute.  Safe for our 1-GPU-per-shard layout (device_map
        # never has to split a single layer across devices).
        from models.modeling_tarsier import TarsierPreTrainedModel  # type: ignore
        if isinstance(getattr(TarsierPreTrainedModel, "_no_split_modules", None), property):
            try:
                delattr(TarsierPreTrainedModel, "_no_split_modules")
            except AttributeError:
                pass
            TarsierPreTrainedModel._no_split_modules = []

        data_config = yaml.safe_load(open(self.CONFIG_PATH))
        data_config["n_frames"] = self.NUM_FRAMES

        # ── HF dynamic-module path workaround ────────────────────────────
        # The Tarsier2 HF config has auto_map entries shaped like
        # "models--modeling_qwen2_vl_fast.Qwen2VLVisionConfig".  Tarsier's
        # LlavaConfig.__init__ splits on "--" to get
        #   repo_id    = "models"
        #   class_ref  = "modeling_qwen2_vl_fast.Qwen2VLVisionConfig"
        # then calls get_class_from_dynamic_module(class_ref, repo_id),
        # which in turn calls cached_files("models", "modeling_qwen2_vl_fast.py").
        # cached_files first checks if "models" is a local directory relative
        # to cwd — and the file *does* exist at
        # third_party/tarsier/models/modeling_qwen2_vl_fast.py.  So we just
        # need cwd to be the tarsier repo root for the duration of load().
        prev_cwd = os.getcwd()
        os.chdir(_TARSIER_REPO)
        try:
            # Patch the dynamically-loaded Qwen2VLPreTrainedModel._init_weights
            # to tolerate Qwen2VLVisionConfig missing `initializer_range`.
            # The dynamic module is only registered after LlavaConfig is built,
            # so we patch lazily by wrapping load_model_and_processor.
            _patch_qwen2vl_dynamic_init_weights()
            # Force AutoImageProcessor to load the SLOW Qwen2VLImageProcessor —
            # Tarsier's per-video tensor branch isinstance-checks against it.
            _patch_tarsier_force_slow_image_processor()
            # load_model_and_processor uses device_map='auto'.
            model, processor = load_model_and_processor(self.MODEL_ID, data_config=data_config)
        finally:
            os.chdir(prev_cwd)
        self.model = model
        self.processor = processor

    def generate_response(self, video_path: str, prompt: str) -> str:
        _load_tarsier_modules()
        from dataset.utils import format_one_sample  # type: ignore

        sample = format_one_sample(video_path, prompt)
        batch = self.processor(sample)
        model_inputs = {
            k: v.to(self.model.device)
            for k, v in batch.items()
            if isinstance(v, torch.Tensor)
        }
        with torch.no_grad():
            outputs = self.model.generate(
                **model_inputs,
                do_sample=False,
                max_new_tokens=self.MAX_NEW_TOKENS,
                top_p=1.0,
                temperature=0.0,
                use_cache=True,
            )
        in_len = model_inputs["input_ids"][0].shape[0]
        text = self.processor.processor.tokenizer.decode(
            outputs[0][in_len:], skip_special_tokens=True
        ).strip()
        return text
