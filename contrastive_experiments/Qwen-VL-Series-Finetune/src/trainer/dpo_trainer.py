"""
QwenDPOTrainer — Direct Preference Optimization for Qwen2/2.5/3 VL models.

This is a self-contained subclass of :class:`transformers.Trainer` so we can
keep full control over the loss, the per-group learning rates (vision / merger
/ LoRA), DeepSpeed ZeRO-2/3 compatibility, and LoRA checkpoint saving — exactly
matching the conventions used by the SFT and contrastive trainers in this repo.

Why we don't subclass :class:`trl.DPOTrainer`:
    * TRL ≥ 1.0 reshapes batches/inputs aggressively (chat templating,
      tokenisation inside ``_prepare_dataset``) and only forwards a fixed
      whitelist of multimodal kwargs to the model. That whitelist excludes
      ``pixel_values_videos`` / ``video_grid_thw`` / ``second_per_grid_ts``,
      which are the keys our Qwen-VL models actually need for video.
    * The TRL trainer also assumes the data collator returns simple
      ``prompt_ids`` / ``chosen_ids`` etc. and rebuilds the batch internally
      — that's redundant given our :class:`DataCollatorForDPODataset`
      already produces the flat ``[chosen; rejected]`` 2*B batch the loss
      needs.

The expected batch (produced by :class:`DataCollatorForDPODataset`)::

    {
        "input_ids":         (2*B, L)  tokens, [chosen; rejected]
        "attention_mask":    (2*B, L)
        "completion_mask":   (2*B, L)  1 on response tokens, 0 on prompt
        "pixel_values_videos": (..., D)  duplicated 2× along batch dim
        "video_grid_thw":    (2*B, 3)
        # optional image equivalents and second_per_grid_ts
    }

Reference log-probabilities are obtained either from an explicit
``ref_model`` (full-fine-tune mode) or by disabling LoRA adapters on the
policy model (PEFT mode).
"""
from __future__ import annotations

import contextlib
import os
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import Trainer
from transformers.trainer import (
    PREFIX_CHECKPOINT_DIR,
    get_parameter_names,
    is_sagemaker_mp_enabled,
    logger,
)
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

from src.train.train_utils import get_peft_state_non_lora_maybe_zero_3


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────
def _selective_log_softmax(logits: torch.Tensor, labels: torch.Tensor,
                           chunk_size: int = 1024) -> torch.Tensor:
    """logp[i, t] = log_softmax(logits[i, t])[labels[i, t]]   (memory-light).

    Processes the sequence dimension in chunks so the intermediate fp32
    tensors for logsumexp/gather never exceed (B, chunk_size, V) — this
    caps peak memory to ~1.2 GB per chunk instead of 37 GB for a 30k-token
    sequence, preventing OOM on long-video DPO batches.
    """
    B, L, V = logits.shape
    out = torch.empty(B, L, dtype=torch.float32, device=logits.device)
    for start in range(0, L, chunk_size):
        end = min(start + chunk_size, L)
        lg = logits[:, start:end, :].float()  # (B, chunk, V) — bf16→fp32 per chunk
        lb = labels[:, start:end]              # (B, chunk)
        lse = torch.logsumexp(lg, dim=-1)     # (B, chunk)
        sel = torch.gather(lg, -1, lb.unsqueeze(-1)).squeeze(-1)  # (B, chunk)
        out[:, start:end] = sel - lse
    return out


@contextlib.contextmanager
def _disable_adapters(peft_model):
    """Context manager: disable LoRA adapters so we get base-model logits."""
    if hasattr(peft_model, "disable_adapter"):
        with peft_model.disable_adapter():
            yield
    else:  # not a PEFT model — caller messed up but be permissive
        yield


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────
# Vision keys we forward to the model (TRL's whitelist misses these).
_VISION_KEYS = (
    "pixel_values",
    "image_grid_thw",
    "image_sizes",
    "pixel_attention_mask",
    "pixel_values_videos",
    "video_grid_thw",
    "second_per_grid_ts",
)


class QwenDPOTrainer(Trainer):
    """Direct-preference trainer for Qwen-VL models."""

    def __init__(
        self,
        model,
        ref_model=None,
        beta: float = 0.1,
        loss_type: str = "sigmoid",
        label_smoothing: float = 0.0,
        is_peft_model: Optional[bool] = None,
        **kwargs,
    ):
        super().__init__(model=model, **kwargs)
        self.ref_model = ref_model
        self.beta = float(beta)
        self.loss_type = loss_type
        self.label_smoothing = float(label_smoothing)
        self._is_peft = (
            is_peft_model
            if is_peft_model is not None
            else hasattr(model, "disable_adapter")
        )
        if self.ref_model is None and not self._is_peft:
            raise ValueError(
                "QwenDPOTrainer needs either an explicit ref_model or a PEFT "
                "policy model (so we can disable adapters for the reference)."
            )
        if self.ref_model is not None:
            # Move ref to the same device, freeze, and let DeepSpeed/Accelerate
            # know about it.  We don't wrap it in DeepSpeed: ref is inference-
            # only and freezing it avoids the optimiser overhead.
            for p in self.ref_model.parameters():
                p.requires_grad_(False)
            self.ref_model.eval()
            try:
                self.ref_model = self.accelerator.prepare_model(
                    self.ref_model, evaluation_mode=True,
                )
            except Exception:  # noqa: BLE001
                pass

        # ── numerical-health diagnostic ('DPO_DIAG_STEPS' env var) ────────
        # When set to N > 0, install forward hooks on the visual tower and
        # merger that report finite-fraction / min / max / mean of the
        # activations on every call for the first N optimizer steps.  This is
        # how we verify whether logit saturation originates in the ViT, the
        # merger, or the LM head.
        self._diag_steps = int(os.environ.get("DPO_DIAG_STEPS", "0"))
        self._diag_call_count = 0  # incremented each _forward_logps call
        self._diag_handles = []
        if self._diag_steps > 0:
            self._install_diag_hooks(model)

        # ── per-parameter NaN-grad sanitiser hooks (lazy install) ────────
        # Installed on first training_step call (after DeepSpeed has wrapped
        # the model and finalised its parameter list).  See _install_grad_
        # nan_hooks() for the rationale.
        self._grad_nan_hooks_installed = False
        self._grad_nan_handles = []
        self._grad_nan_count = 0  # per-param-per-microbatch fire count

    # ── diagnostic hooks ──────────────────────────────────────────
    @staticmethod
    def _tensor_health(name: str, t: torch.Tensor, prefix: str = "") -> str:
        if not isinstance(t, torch.Tensor):
            return f"{prefix}{name}: not-a-tensor (type={type(t).__name__})"
        with torch.no_grad():
            tt = t.detach()
            finite = torch.isfinite(tt)
            n_total = tt.numel()
            n_finite = int(finite.sum().item())
            if n_finite == 0:
                return (f"{prefix}{name}: shape={tuple(tt.shape)} dtype={tt.dtype} "
                        f"finite=0/{n_total}")
            tt_finite = tt[finite].float()
            return (
                f"{prefix}{name}: shape={tuple(tt.shape)} dtype={tt.dtype} "
                f"finite={n_finite}/{n_total} "
                f"min={tt_finite.min().item():+.3e} "
                f"max={tt_finite.max().item():+.3e} "
                f"mean={tt_finite.mean().item():+.3e} "
                f"absmax={tt_finite.abs().max().item():+.3e}"
            )

    def _install_diag_hooks(self, model) -> None:
        """Attach forward hooks on visual tower + merger for the first N steps."""
        # Resolve the actual module tree.  After PEFT wrapping the path is
        # model.base_model.model.visual; bare model is model.visual.
        roots = []
        for m in (model, getattr(model, "base_model", None)):
            if m is None:
                continue
            inner = getattr(m, "model", m)
            if hasattr(inner, "visual"):
                roots.append(("visual", inner.visual))
                if hasattr(inner.visual, "merger"):
                    roots.append(("visual.merger", inner.visual.merger))
                break

        if not roots:
            logger.warning("DPO_DIAG: could not locate model.visual; hooks not installed")
            return

        def _make_hook(tag: str):
            def _hook(_module, _inp, out):
                if self._diag_call_count > self._diag_steps:
                    return
                t = out[0] if isinstance(out, (tuple, list)) else out
                msg = self._tensor_health(tag, t,
                                          prefix=f"  [DPO_DIAG call#{self._diag_call_count}] ")
                logger.warning(msg)
            return _hook

        for tag, mod in roots:
            self._diag_handles.append(mod.register_forward_hook(_make_hook(tag)))
        logger.warning(f"DPO_DIAG: installed forward hooks on {[t for t,_ in roots]} "
                       f"for first {self._diag_steps} calls")

    # ── log helper ─────────────────────────────────────────────────
    def _store_metrics(self, metrics: Dict[str, float]) -> None:
        """Buffer a metrics dict so it gets averaged into the next .log() call."""
        if not hasattr(self, "_dpo_metric_buffer"):
            self._dpo_metric_buffer: Dict[str, list] = {}
        for k, v in metrics.items():
            self._dpo_metric_buffer.setdefault(k, []).append(float(v))

    def log(self, logs, *args, **kwargs):  # type: ignore[override]
        if hasattr(self, "_dpo_metric_buffer") and self._dpo_metric_buffer:
            for k, vs in self._dpo_metric_buffer.items():
                # NaN-safe: drop NaN/inf samples before averaging
                clean = [v for v in vs if v == v and v not in (float("inf"), float("-inf"))]
                if clean:
                    logs[k] = sum(clean) / len(clean)
            self._dpo_metric_buffer = {}
        return super().log(logs, *args, **kwargs)

    # ── main loss ────────────────────────────────────────────────────────
    def _forward_logps(self, model, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward through ``model`` and compute per-sequence log p(completion)."""
        input_ids: torch.Tensor = inputs["input_ids"]
        attention_mask: torch.Tensor = inputs["attention_mask"]
        completion_mask: torch.Tensor = inputs["completion_mask"]

        model_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": False,
        }
        for k in _VISION_KEYS:
            if k in inputs:
                model_kwargs[k] = inputs[k]

        outputs = model(**model_kwargs)

        # ── finite check on the raw bf16 logits (before fp32 cast) ────────
        # Only check for NaN, not inf — bf16 max is 65504 so large-but-finite
        # fp32 logits appear as inf in bf16.  That's harmless; NaN is the real
        # poison that corrupts gradients.
        raw_logits = outputs.logits
        with torch.no_grad():
            has_nan = bool(torch.isnan(raw_logits).any().item())
        logits_all_finite = not has_nan

        # ── diagnostic: report logits health for the first N calls ────────
        if self._diag_steps > 0 and self._diag_call_count <= self._diag_steps:
            logger.warning(self._tensor_health(
                "lm_logits(bf16)", raw_logits,
                prefix=f"  [DPO_DIAG call#{self._diag_call_count}] ",
            ))
        self._diag_call_count += 1

        # Cast to float32 for the log-softmax — but do NOT materialise the
        # full (2B, L, V) fp32 tensor here.  Instead keep bf16 logits and let
        # _selective_log_softmax cast per-chunk, capping peak memory.
        logits = raw_logits          # stay bf16 — chunked cast below
        del raw_logits

        # Standard next-token: predict input_ids[t+1] from logits[t]
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = completion_mask[:, 1:].contiguous().float()

        per_tok_logps = _selective_log_softmax(shift_logits, shift_labels)
        del shift_logits, logits  # free the big bf16 tensor ASAP
        # Hard-zero pad/non-completion positions before sum to avoid NaN
        # propagation if the model emits inf logits at unused positions.
        per_tok_logps = torch.where(
            shift_mask > 0, per_tok_logps, torch.zeros_like(per_tok_logps)
        )
        # Defensive: nan_to_num so a single bad logit can't poison the whole
        # batch's metrics.  The loss path is independent (it sums then reduces).
        per_tok_logps = torch.nan_to_num(per_tok_logps, nan=0.0,
                                         posinf=0.0, neginf=0.0)
        seq_logps = per_tok_logps.sum(dim=-1)
        seq_lens = shift_mask.sum(dim=-1).clamp(min=1.0)
        return seq_logps, seq_lens, shift_mask, logits_all_finite

    def _dpo_loss(
        self,
        chosen_logps: torch.Tensor,
        rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
        chosen_lens: torch.Tensor,
        rejected_lens: torch.Tensor,
    ) -> torch.Tensor:
        chosen_logratios = chosen_logps - ref_chosen_logps
        rejected_logratios = rejected_logps - ref_rejected_logps
        logits = chosen_logratios - rejected_logratios

        if self.loss_type == "sigmoid":
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        elif self.loss_type == "ipo":
            # IPO uses length-normalised logp differences.
            chosen_avg = chosen_logratios / chosen_lens
            rejected_avg = rejected_logratios / rejected_lens
            losses = (chosen_avg - rejected_avg - 1 / (2 * self.beta)) ** 2
        else:
            raise ValueError(f"Unknown DPO loss_type: {self.loss_type}")
        return losses.mean()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):  # type: ignore[override]
        # ── policy logps ────────────────────────────────────────────────
        seq_logps, seq_lens, _, policy_finite = self._forward_logps(model, inputs)

        # If the policy forward produced any non-finite logit we MUST not let
        # the gradient flow back through the autograd graph — it would write
        # NaN onto every trainable LoRA weight and permanently corrupt the
        # model.  Build a synthetic loss that depends on every trainable param
        # via 0 * sum(p) so DDP all-reduce produces zero grads on every rank
        # (i.e. effectively skip the optimizer step for this batch).
        if not policy_finite:
            self._store_metrics({"train/skipped_steps": 1.0})
            _skip_count = getattr(self, "_nan_skip_count", 0) + 1
            self._nan_skip_count = _skip_count
            if _skip_count <= 5 or _skip_count % 50 == 0:
                logger.warning(
                    f"DPO: NaN in policy logits — skipping optimizer step "
                    f"(occurrence {_skip_count})."
                )
            zero_loss = sum(
                (p * 0.0).sum() for p in model.parameters() if p.requires_grad
            )
            # Wrap into a tensor with grad tracking; mean of zero is zero but
            # autograd still sees the parameter dependency.
            loss = zero_loss + 0.6931471805599453  # report log(2) for clarity
            with torch.no_grad():
                self._store_metrics({
                    "rewards/chosen": 0.0, "rewards/rejected": 0.0,
                    "rewards/margin": 0.0, "rewards/accuracy": 0.0,
                    "logps/chosen": float("nan"), "logps/rejected": float("nan"),
                    "logps/ref_chosen": float("nan"), "logps/ref_rejected": float("nan"),
                })
            return (loss, {}) if return_outputs else loss

        chosen_logps, rejected_logps = seq_logps.chunk(2, dim=0)
        chosen_lens, rejected_lens = seq_lens.chunk(2, dim=0)

        # ── reference logps ─────────────────────────────────────────────
        with torch.no_grad():
            if self.ref_model is not None:
                ref_seq_logps, _, _, _ = self._forward_logps(self.ref_model, inputs)
            else:
                # PEFT: disable adapters → base model
                with _disable_adapters(model):
                    ref_seq_logps, _, _, _ = self._forward_logps(model, inputs)
            ref_seq_logps = torch.nan_to_num(ref_seq_logps, nan=0.0,
                                             posinf=0.0, neginf=0.0)
        ref_chosen_logps, ref_rejected_logps = ref_seq_logps.chunk(2, dim=0)

        loss = self._dpo_loss(
            chosen_logps, rejected_logps,
            ref_chosen_logps, ref_rejected_logps,
            chosen_lens, rejected_lens,
        )
        # Final safety net: if loss itself is non-finite, swap to the same
        # zero-grad path used above.
        if not torch.isfinite(loss):
            self._store_metrics({"train/skipped_steps": 1.0})
            zero_loss = sum(
                (p * 0.0).sum() for p in model.parameters() if p.requires_grad
            )
            loss = zero_loss + 0.6931471805599453

        # ── metrics ─────────────────────────────────────────────────────
        with torch.no_grad():
            chosen_rewards = self.beta * (chosen_logps - ref_chosen_logps)
            rejected_rewards = self.beta * (rejected_logps - ref_rejected_logps)
            self._store_metrics({
                "rewards/chosen": chosen_rewards.mean().item(),
                "rewards/rejected": rejected_rewards.mean().item(),
                "rewards/margin": (chosen_rewards - rejected_rewards).mean().item(),
                "rewards/accuracy": (chosen_rewards > rejected_rewards).float().mean().item(),
                "logps/chosen": chosen_logps.mean().item(),
                "logps/rejected": rejected_logps.mean().item(),
                "logps/ref_chosen": ref_chosen_logps.mean().item(),
                "logps/ref_rejected": ref_rejected_logps.mean().item(),
            })

        if return_outputs:
            return loss, {
                "chosen_logps": chosen_logps,
                "rejected_logps": rejected_logps,
            }
        return loss

    # ── eval: route through DPO loss instead of HF default ───────────────
    # Trainer.prediction_step falls back to model(**inputs) when there is no
    # `labels` key — that materialises full (2, L, V) logits in eval and
    # never computes the DPO loss.  We override to call compute_loss in
    # no_grad mode and return the scalar loss + None outputs/labels (we have
    # no token-level labels in DPO; the rewards/* metrics are stored via
    # _store_metrics inside compute_loss and surfaced in the next .log()).
    def prediction_step(self, model, inputs, prediction_loss_only,
                        ignore_keys=None):  # type: ignore[override]
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss = self.compute_loss(model, inputs, return_outputs=False)
        loss = loss.detach()
        return (loss, None, None)

    # ── per-parameter NaN-grad sanitiser ─────────────────────────────────
    # PROBLEM: bf16 backward through long video sequences (65k tokens, 2×
    # forwards for policy+ref) can produce NaN/inf in the gradient of an
    # individual parameter even when the loss itself was finite.  Under
    # gradient accumulation, a single such bad microbatch poisons the entire
    # window's accumulated grad buffer; the optimizer then steps on NaN, the
    # LoRA weights become NaN, and every subsequent forward returns NaN
    # logits forever.  This is exactly what we observed in production
    # (occurrence count climbing 1750 → 1800 → 1850 with loss stuck at 0).
    #
    # FIX: register a backward hook on each trainable parameter via
    # ``Tensor.register_hook(fn)``.  The hook receives the parameter's
    # gradient *before* it is accumulated into ``.grad`` (and before any
    # DeepSpeed reduce-scatter), and may return a replacement tensor.  We
    # use ``torch.nan_to_num`` to zero out any NaN/inf elements in-place.
    #
    # Why this is correct under DeepSpeed ZeRO-2:
    # - Backward hooks fire DURING autograd, on the locally-computed
    #   gradient tensor on the active rank.  This is upstream of any
    #   communication or accumulation that DeepSpeed performs.
    # - Replacing NaN with 0 means the bad microbatch contributes ZERO to
    #   the accumulation buffer (a no-op, not a corruption).  Other clean
    #   microbatches in the same window still contribute their real grads,
    #   and the optimizer steps on a valid (if slightly smaller) batch.
    # - The previous snapshot/restore approach in training_step was silently
    #   broken under ZeRO-2: ``p.grad`` is often ``None`` post-backward (the
    #   gradient lives in DeepSpeed's flat ``ipg_buffer``), so our
    #   ``if p.grad is None: continue`` skip never detected NaN.
    def _install_grad_nan_hooks(self, model) -> None:
        if self._grad_nan_hooks_installed:
            return
        # Counter dict shared across all hook closures (per-parameter fire
        # count).  Reported via _store_metrics on the next log() call.
        counter = {"fires": 0, "params_hit": 0}
        self._grad_nan_counter = counter

        def _make_hook(pname: str):
            def _hook(grad: torch.Tensor):
                # Fast-path: ``isnan().any() | isinf().any()`` is one fused
                # reduction and stays on-device.
                with torch.no_grad():
                    bad = torch.isnan(grad).any() | torch.isinf(grad).any()
                if bool(bad):
                    counter["fires"] += 1
                    counter["params_hit"] += 1
                    if counter["fires"] <= 5 or counter["fires"] % 200 == 0:
                        logger.warning(
                            f"DPO grad-nan: zeroing NaN/inf in grad of "
                            f"{pname} (fire #{counter['fires']})"
                        )
                    # Replace bad elements with 0; preserve dtype/device.
                    return torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
                return grad  # unchanged
            return _hook

        n = 0
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self._grad_nan_handles.append(p.register_hook(_make_hook(name)))
            n += 1
        self._grad_nan_hooks_installed = True
        logger.warning(
            f"DPO: installed NaN-grad sanitiser hooks on {n} trainable params"
        )

    def training_step(self, model, inputs, num_items_in_batch=None):  # type: ignore[override]
        # Lazy install on first call: by now PEFT wrapping + DeepSpeed engine
        # creation are done, and ``model.named_parameters()`` returns the
        # final, post-wrap parameter list.
        if not self._grad_nan_hooks_installed:
            self._install_grad_nan_hooks(model)

        loss = super().training_step(model, inputs, num_items_in_batch)

        # Surface the per-microbatch fire count, then reset.
        c = getattr(self, "_grad_nan_counter", None)
        if c is not None and c["fires"] > 0:
            self._store_metrics({
                "train/grad_nan_fires": float(c["fires"]),
                "train/grad_nan_params_hit": float(c["params_hit"]),
            })
            c["fires"] = 0
            c["params_hit"] = 0
        return loss

    # ── group-LR optimiser (mirrors SFT trainer) ─────────────────────────
    def create_optimizer(self):
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()
        opt_model = self.model
        if self.optimizer is not None:
            return self.optimizer

        decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [n for n in decay_parameters if "bias" not in n]

        vision_lr = getattr(self.args, "vision_lr", None)
        merger_lr = getattr(self.args, "merger_lr", None)

        visual_params = (
            [n for n, _ in opt_model.named_parameters() if "visual" in n and "merger" not in n]
            if vision_lr is not None else []
        )
        merger_params = (
            [n for n, _ in opt_model.named_parameters() if "merger" in n]
            if merger_lr is not None else []
        )
        special = set(visual_params) | set(merger_params)

        groups = [
            {"params": [p for n, p in opt_model.named_parameters()
                        if n in decay_parameters and n not in special and p.requires_grad],
             "weight_decay": self.args.weight_decay},
            {"params": [p for n, p in opt_model.named_parameters()
                        if n not in decay_parameters and n not in special and p.requires_grad],
             "weight_decay": 0.0},
        ]
        if visual_params:
            groups += [
                {"params": [p for n, p in opt_model.named_parameters()
                            if n in decay_parameters and n in visual_params and p.requires_grad],
                 "weight_decay": self.args.weight_decay, "lr": vision_lr,
                 "param_group_name": "visual_decay"},
                {"params": [p for n, p in opt_model.named_parameters()
                            if n not in decay_parameters and n in visual_params and p.requires_grad],
                 "weight_decay": 0.0, "lr": vision_lr,
                 "param_group_name": "visual_no_decay"},
            ]
        if merger_params:
            groups += [
                {"params": [p for n, p in opt_model.named_parameters()
                            if n in decay_parameters and n in merger_params and p.requires_grad],
                 "weight_decay": self.args.weight_decay, "lr": merger_lr,
                 "param_group_name": "merger_decay"},
                {"params": [p for n, p in opt_model.named_parameters()
                            if n not in decay_parameters and n in merger_params and p.requires_grad],
                 "weight_decay": 0.0, "lr": merger_lr,
                 "param_group_name": "merger_no_decay"},
            ]

        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(groups, **optimizer_kwargs)
        return self.optimizer

    # ── checkpoint extra: save non-LoRA trainables ───────────────────────
    def _save_checkpoint(self, model, trial):  # type: ignore[override]
        super()._save_checkpoint(model, trial)
        if not getattr(self.args, "lora_enable", False):
            return
        ckpt_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        run_dir = self._get_output_dir(trial=trial)
        out_dir = os.path.join(run_dir, ckpt_folder)
        non_lora = get_peft_state_non_lora_maybe_zero_3(
            self.model.named_parameters(), require_grad_only=True,
        )
        if self.args.should_save:
            torch.save(non_lora, os.path.join(out_dir, "non_lora_state_dict.bin"))
            try:
                self.model.base_model.config.to_json_file(
                    os.path.join(out_dir, "config.json")
                )
            except Exception:  # noqa: BLE001
                pass
