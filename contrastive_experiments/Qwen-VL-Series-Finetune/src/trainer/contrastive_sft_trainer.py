"""
==============================================================================
Contrastive SFT Trainer
==============================================================================
Extends QwenSFTTrainer with a contrastive regularization loss:

    L_total = L_next_token + λ · L_regularizer

Where L_regularizer is an InfoNCE loss that:
  - Pulls the model toward grounding answers in actual visual context
  - Pushes it away from pre-trained knowledge hallucinations

Two CONTRASTIVE_MODE options:
  "generative" — compute standard next-token cross-entropy loss on the full
     [Video+Question+Answer] sequence, contrast the log-likelihood of the
     true video against corrupted/other videos.
  "vector" — run separate forward passes with output_hidden_states=True to
     extract the final [EOS] hidden state vectors, project through a linear
     layer, compute cosine similarity (following the Amazon paper "Aligning
     Vision Language Models with Contrastive Learning").

Supports experiments V-01..V-05, T-01..T-03, FULL, and CUSTOM.

Stage: Training Loop & Loss Computation
==============================================================================
"""

import os
import random
import math
import contextlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

from transformers import Trainer, GenerationConfig
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    TRAINER_STATE_NAME,
    PREFIX_CHECKPOINT_DIR,
    logger,
    has_length,
)
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_utils import EvalLoopOutput

from src.train.train_utils import (
    get_peft_state_maybe_zero_3,
    get_peft_state_non_lora_maybe_zero_3,
)
from src.constants import IGNORE_INDEX
from src.contrastive_utils import (
    blacken_pixel_values,
    gaussianize_pixel_values,
    compute_temporal_shift,
    compute_multiple_temporal_shifts,
    build_in_batch_negative_indices,
    compute_infonce_loss,
    compute_generation_log_likelihood,
    compute_generation_log_likelihood_masked,
    compute_generation_log_likelihood_weighted,
    compute_distractor_negative_score,
    build_entity_token_mask,
    build_anchor_token_weights,
    extract_eos_hidden_states,
    get_experiment_config,
    ContrastiveProjectionHead,
)


# ═══════════════════════════════════════════════════════════════════════
# SourceBlockSampler — keeps every batch single-source, random block order
# (Same as vanilla SFT trainer — reused for consistency)
# ═══════════════════════════════════════════════════════════════════════

class SourceBlockSampler(Sampler):
    """
    Sampler that yields indices in blocks where every block contains samples
    from a single source. Block order is shuffled so each source has
    ~uniform probability of appearing next.
    
    This is critical for contrastive learning: in-batch negatives only make
    sense when all samples in a batch share the same domain/source.
    """

    def __init__(self, dataset, batch_size: int, seed: int = 42,
                 rank: int = 0, world_size: int = 1):
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0

        self.source_to_indices: Dict[str, List[int]] = defaultdict(list)
        for idx, item in enumerate(dataset.list_data_dict):
            src = item.get("source", "unknown")
            self.source_to_indices[src].append(idx)

        logger.info(
            f"SourceBlockSampler (CL): {len(dataset)} samples, "
            f"{len(self.source_to_indices)} sources: "
            + ", ".join(
                f"{k}={len(v)}" for k, v in sorted(self.source_to_indices.items())
            )
        )

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        g = random.Random(self.seed + self.epoch)
        all_blocks = []
        for src, indices in self.source_to_indices.items():
            idxs = indices.copy()
            g.shuffle(idxs)
            for start in range(0, len(idxs) - self.batch_size + 1, self.batch_size):
                all_blocks.append(idxs[start : start + self.batch_size])
        g.shuffle(all_blocks)
        if self.world_size > 1:
            total_blocks = len(all_blocks)
            blocks_per_rank = total_blocks // self.world_size
            all_blocks = all_blocks[: blocks_per_rank * self.world_size]
            all_blocks = all_blocks[self.rank :: self.world_size]
        for block in all_blocks:
            yield from block

    def __len__(self):
        total = sum(
            (len(v) // self.batch_size) * self.batch_size
            for v in self.source_to_indices.values()
        )
        if self.world_size > 1:
            total = total // self.world_size
        return total


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, "no ignore status")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


@dataclass
class GenerativeEvalPrediction:
    """Container for generative evaluation predictions."""
    predictions: List[str]
    references: List[str]


# ═══════════════════════════════════════════════════════════════════════
# ContrastiveSFTTrainer — the main trainer class
# ═══════════════════════════════════════════════════════════════════════

class ContrastiveSFTTrainer(Trainer):
    """
    Trainer that adds contrastive regularization to standard SFT.
    
    Key stages:
      1. Standard forward pass → L_next_token
      2. Generate negatives (in-batch, blackened, gaussian, temporal)
      3. Compute similarity scores (generative or vector mode)
      4. Compute L_regularizer via InfoNCE
      5. L_total = L_next_token + λ · L_regularizer
    
    The λ is set to 0 for FineVideo samples (no timestamps → no grounding
    possible), preserving long-context general QA abilities.
    """

    def __init__(
        self,
        *args,
        contrastive_args=None,
        projection_head: Optional[ContrastiveProjectionHead] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # ── Store contrastive configuration ──
        self.cl_args = contrastive_args
        self.projection_head = projection_head

        # ── Gradient context for negative forward passes ──
        self._grad_through_negs = (
            getattr(contrastive_args, "grad_through_negatives", False)
            if contrastive_args is not None else False
        )

        # ── Resolve experiment config ──
        if self.cl_args is not None:
            self.experiment_config = get_experiment_config(
                self.cl_args.negative_strategy
            )
            # Allow CLI overrides for alpha
            if self.cl_args.alpha_grounding_penalty is not None:
                self.experiment_config["default_alpha"] = (
                    self.cl_args.alpha_grounding_penalty
                )
        else:
            self.experiment_config = get_experiment_config("V-01")

        # ── Contrastive step counter for logging ──
        self._cl_step = 0

    # ═══════════════════════════════════════════════════════════════
    # Stage: DataLoader (same as vanilla — single-source batches)
    # ═══════════════════════════════════════════════════════════════

    def get_train_dataloader(self) -> DataLoader:
        """
        Override to use SourceBlockSampler so every batch is single-source.
        This is essential for meaningful in-batch contrastive learning.
        """
        dataset = self.train_dataset
        data_collator = self.data_collator
        world_size = self.args.world_size if self.args.world_size else 1
        rank = self.args.process_index if self.args.process_index else 0

        sampler = SourceBlockSampler(
            dataset=dataset,
            batch_size=self.args.per_device_train_batch_size,
            seed=self.args.seed if self.args.seed else 42,
            rank=rank,
            world_size=world_size,
        )

        return DataLoader(
            dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            drop_last=self.args.dataloader_drop_last,
        )

    # ═══════════════════════════════════════════════════════════════
    # Stage: Optimizer (supports separate LRs for vision/merger)
    # ═══════════════════════════════════════════════════════════════

    def create_optimizer(self):
        """
        Same optimizer setup as vanilla SFT: supports separate LRs for
        vision tower and merger. Additionally includes the projection head
        parameters (if vector mode is used).
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [
                name for name in decay_parameters if "bias" not in name
            ]
            lr_mapper = {}
            visual_parameters = []
            merger_parameters = []

            if self.args.vision_lr is not None:
                lr_mapper["visual"] = self.args.vision_lr
                visual_parameters = [
                    name
                    for name, _ in opt_model.named_parameters()
                    if "visual" in name and "merger" not in name
                ]
            if self.args.merger_lr is not None:
                lr_mapper["merger"] = self.args.merger_lr
                merger_parameters = [
                    name
                    for name, _ in opt_model.named_parameters()
                    if "merger" in name
                ]

            if len(lr_mapper) > 0:
                special_lr_parameters = merger_parameters + visual_parameters

                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n not in special_lr_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n not in special_lr_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                    },
                ]

                if visual_parameters:
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [
                                    p
                                    for n, p in opt_model.named_parameters()
                                    if (
                                        n in decay_parameters
                                        and n in visual_parameters
                                        and p.requires_grad
                                    )
                                ],
                                "weight_decay": self.args.weight_decay,
                                "lr": self.args.vision_lr,
                            },
                            {
                                "params": [
                                    p
                                    for n, p in opt_model.named_parameters()
                                    if (
                                        n not in decay_parameters
                                        and n in visual_parameters
                                        and p.requires_grad
                                    )
                                ],
                                "weight_decay": 0.0,
                                "lr": self.args.vision_lr,
                            },
                        ]
                    )

                if merger_parameters:
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [
                                    p
                                    for n, p in opt_model.named_parameters()
                                    if (
                                        n in decay_parameters
                                        and n in merger_parameters
                                        and p.requires_grad
                                    )
                                ],
                                "weight_decay": self.args.weight_decay,
                                "lr": self.args.merger_lr,
                            },
                            {
                                "params": [
                                    p
                                    for n, p in opt_model.named_parameters()
                                    if (
                                        n not in decay_parameters
                                        and n in merger_parameters
                                        and p.requires_grad
                                    )
                                ],
                                "weight_decay": 0.0,
                                "lr": self.args.merger_lr,
                            },
                        ]
                    )
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            # Stage: Projection head parameters (vector mode only)
            # NOTE: The projection head is registered as a submodule of the
            # model (model.contrastive_projection_head) so DeepSpeed can
            # track it.  Its parameters already appear in
            # opt_model.named_parameters() and are picked up by the decay /
            # no-decay groups above.  No need to add them explicitly.

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                self.args
            )
            self.optimizer = optimizer_cls(
                optimizer_grouped_parameters, **optimizer_kwargs
            )

            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()
                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum(
                            {
                                p.data_ptr(): p.numel()
                                for p in module.parameters()
                            }.values()
                        )
                        logger.info(f"skipped {module}: {skipped / 2**20}M params")
                        manager.register_module_override(
                            module, "weight", {"optim_bits": 32}
                        )
                        logger.debug(
                            f"bitsandbytes: will optimize {module} in fp32"
                        )
                logger.info(f"skipped: {skipped / 2**20}M params")

        return self.optimizer

    # ═══════════════════════════════════════════════════════════════
    # Stage: Checkpoint saving (save LoRA + non-LoRA state dicts)
    # ═══════════════════════════════════════════════════════════════

    def _save_checkpoint(self, model, trial):
        super()._save_checkpoint(model, trial)

        if not self.args.lora_enable:
            return

        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)

        non_lora = get_peft_state_non_lora_maybe_zero_3(
            self.model.named_parameters(),
            require_grad_only=True,
        )

        if self.args.should_save:
            torch.save(
                non_lora,
                os.path.join(output_dir, "non_lora_state_dict.bin"),
            )
            self.model.base_model.config.to_json_file(
                os.path.join(output_dir, "config.json")
            )

            # Stage: Also save projection head if it exists (vector mode)
            if self.projection_head is not None:
                torch.save(
                    self.projection_head.state_dict(),
                    os.path.join(output_dir, "projection_head.bin"),
                )

    # ═══════════════════════════════════════════════════════════════
    # Stage: Core — compute_loss override with contrastive regularizer
    # ═══════════════════════════════════════════════════════════════

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Main training loss:
            L_total = L_next_token + λ · L_contrastive
        
        Steps:
          1. Pop cl_metadata from inputs (not a model input)
          2. Forward pass → standard SFT cross-entropy loss (L_next_token)
          3. If CL is enabled and batch is CL-eligible:
             a. Generate negative variants (blackened, gaussian, temporal, batch)
             b. Compute similarity scores via chosen CONTRASTIVE_MODE
             c. Compute InfoNCE loss
             d. Apply per-sample λ weighting (0 for FineVideo)
          4. Return L_total
        """
        # ── Step 1: Pop non-tensor metadata before model forward ──
        cl_metadata = inputs.pop("cl_metadata", None)

        # ── Step 2: Standard SFT forward pass ──
        # For vector contrastive mode, we register a forward hook on the
        # inner Qwen3VLModel BEFORE the main forward so we capture
        # last_hidden_state during the SAME forward pass that all ranks
        # participate in. This avoids NCCL hangs from an extra model
        # forward that only CL-eligible ranks would execute.
        _captured_hidden = {}
        _hook_handle = None
        if (
            self.cl_args is not None
            and getattr(self.cl_args, "contrastive_mode", None) == "vector"
            and self.projection_head is not None
        ):
            inner_model = self._find_inner_model(model)
            if inner_model is not None:
                def _capture_hook(module, input, output):
                    if hasattr(output, "last_hidden_state"):
                        _captured_hidden["lhs"] = output.last_hidden_state
                    else:
                        _captured_hidden["lhs"] = output[0]
                _hook_handle = inner_model.register_forward_hook(_capture_hook)

        outputs = model(**inputs)

        if _hook_handle is not None:
            _hook_handle.remove()

        sft_loss = outputs.loss  # Standard next-token cross-entropy

        # ── Vector mode: anchor projection_head into the loss graph ──
        # In multi-GPU training, ALL ranks must produce gradients for ALL
        # parameters (for DeepSpeed's gradient all-reduce). The projection_head
        # is only used in the CL path, which some ranks may skip (non-eligible
        # batches). Adding a zero-valued term ensures projection_head params
        # always appear in the backward graph, preventing NCCL deadlocks.
        _is_vector_mode = (
            self.cl_args is not None
            and getattr(self.cl_args, "contrastive_mode", None) == "vector"
            and self.projection_head is not None
        )
        if _is_vector_mode:
            proj_anchor = sum(p.sum() for p in self.projection_head.parameters()) * 0.0
            sft_loss = sft_loss + proj_anchor

        # ── Step 3: Contrastive regularization ──
        # NOTE: The CL path runs an extra forward pass on corrupted inputs
        # under torch.no_grad(), then computes InfoNCE from log-likelihoods.
        # To handle variable computation graphs (CL-eligible vs not) with
        # DeepSpeed ZeRO, we use zero1_cl.json which sets:
        #   overlap_comm=false, contiguous_gradients=false
        # This avoids the IPG bucket IndexError without needing a logits
        # anchor hack (which conflicts with Liger kernel's fused CE).
        if self.cl_args is None or cl_metadata is None:
            return (sft_loss, outputs) if return_outputs else sft_loss

        contrastive_weight = self.cl_args.contrastive_weight
        if contrastive_weight <= 0:
            return (sft_loss, outputs) if return_outputs else sft_loss

        # Check if any sample in this batch is CL-eligible
        eligible_mask = [m.get("cl_eligible", False) for m in cl_metadata]
        if not any(eligible_mask):
            return (sft_loss, outputs) if return_outputs else sft_loss

        # ── Compute contrastive loss based on CONTRASTIVE_MODE ──
        contrastive_mode = self.cl_args.contrastive_mode
        try:
            if contrastive_mode == "generative":
                cl_loss = self._compute_generative_contrastive_loss(
                    model, inputs, outputs, cl_metadata
                )
            elif contrastive_mode == "vector":
                cl_loss = self._compute_vector_contrastive_loss(
                    model, inputs, outputs, cl_metadata,
                    cached_hidden_state=_captured_hidden.get("lhs"),
                )
            else:
                raise ValueError(
                    f"Unknown contrastive_mode: {contrastive_mode}. "
                    f"Must be 'generative' or 'vector'."
                )
        except Exception as e:
            # If CL loss computation fails, log and fall back to pure SFT
            if self._cl_step % 50 == 0:
                logger.warning(
                    f"CL loss computation failed at step {self._cl_step}: {e}. "
                    f"Falling back to pure SFT loss."
                )
            cl_loss = torch.tensor(0.0, device=sft_loss.device, dtype=sft_loss.dtype)

        # ── Step 4: Combine losses ──
        eligible_fraction = sum(eligible_mask) / len(eligible_mask)
        effective_lambda = contrastive_weight * eligible_fraction

        # Guard against NaN/Inf in CL loss (can happen with degenerate batches)
        if torch.isnan(cl_loss) or torch.isinf(cl_loss):
            cl_loss = torch.tensor(0.0, device=sft_loss.device, dtype=sft_loss.dtype)

        total_loss = sft_loss + effective_lambda * cl_loss

        # Final NaN guard — if sft_loss is NaN (e.g. all labels truncated),
        # skip this step by returning zero loss.  The step is wasted but
        # training continues without corrupting parameters.
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            if self._cl_step % 20 == 0:
                logger.warning(
                    f"NaN/Inf total_loss at CL step {self._cl_step}. "
                    f"Skipping update (zero loss)."
                )
            total_loss = torch.tensor(0.0, device=sft_loss.device,
                                      dtype=sft_loss.dtype, requires_grad=True)

        # ── Logging ──
        self._cl_step += 1
        if self.cl_args.log_contrastive_metrics and self._cl_step % 10 == 0:
            self._log_cl_metrics(sft_loss, cl_loss, total_loss, eligible_fraction)

        return (total_loss, outputs) if return_outputs else total_loss

    # ═══════════════════════════════════════════════════════════════
    # Stage: Generative contrastive mode
    # ═══════════════════════════════════════════════════════════════

    def _compute_generative_contrastive_loss(
        self,
        model,
        inputs: Dict[str, torch.Tensor],
        positive_outputs,
        cl_metadata: List[dict],
    ) -> torch.Tensor:
        """
        Generative contrastive mode:
        
        s(V, T) = average per-token log-probability of answer given [V+Q]
        
        For each sample i in the batch:
          - Positive: s(V_i, T_i) — already computed in the forward pass
          - Batch negatives: s(V_j, T_i) for j ≠ i — use other batch items' pixels
          - Grounding negatives: s(V~_i, T_i) — blackened/gaussian/temporal
        
        Memory efficiency: we reuse the batch's forward passes. For each
        sample, the "positive score" is its own log-likelihood. The
        "batch negative score" is the log-likelihood of OTHER samples'
        answers given THIS sample's video (or vice versa, depending on
        experiment type).
        """
        device = positive_outputs.logits.device
        batch_size = inputs["input_ids"].shape[0]
        exp_cfg = self.experiment_config
        temperature = self.cl_args.contrastive_temperature
        alpha = exp_cfg["default_alpha"]
        use_entity_masking = exp_cfg.get("use_entity_masking", False)
        use_anchor_weighting = exp_cfg.get("use_anchor_weighting", False)
        anchor_weight = exp_cfg.get("anchor_weight", 5.0)

        # ── Build entity masks if V-06 entity masking is active ──
        entity_masks = None
        if use_entity_masking:
            tokenizer = self.processing_class.tokenizer
            entity_masks = []
            for i in range(batch_size):
                em = build_entity_token_mask(
                    inputs["labels"][i], tokenizer, ignore_index=IGNORE_INDEX,
                )
                entity_masks.append(em)

        # ── Build anchor-weight tensors if V-07 anchor weighting is active ──
        anchor_weights = None
        if use_anchor_weighting:
            tokenizer = self.processing_class.tokenizer
            anchor_weights = []
            for i in range(batch_size):
                anchor_text = ""
                if cl_metadata is not None and i < len(cl_metadata):
                    anchor_text = cl_metadata[i].get("gold_anchor_text", "") or ""
                w = build_anchor_token_weights(
                    inputs["labels"][i], anchor_text, tokenizer,
                    anchor_weight=anchor_weight, base_weight=1.0,
                    ignore_index=IGNORE_INDEX,
                )
                anchor_weights.append(w)

        # ── Compute positive scores: log-likelihood per sample ──
        positive_scores = []
        for i in range(batch_size):
            if anchor_weights is not None:
                score = compute_generation_log_likelihood_weighted(
                    positive_outputs.logits[i],
                    inputs["labels"][i],
                    anchor_weights[i],
                    ignore_index=IGNORE_INDEX,
                )
            elif entity_masks is not None:
                score = compute_generation_log_likelihood_masked(
                    positive_outputs.logits[i],
                    inputs["labels"][i],
                    entity_masks[i],
                    ignore_index=IGNORE_INDEX,
                )
            else:
                score = compute_generation_log_likelihood(
                    positive_outputs.logits[i],
                    inputs["labels"][i],
                    ignore_index=IGNORE_INDEX,
                )
            positive_scores.append(score)
        positive_scores = torch.stack(positive_scores)  # [B]

        # ── Compute batch negative scores (V-01 / T-01 style) ──
        # For each sample i, compute s(V_j, T_i) = log-likelihood of
        # sample i's answer given sample j's video.  This requires a
        # separate forward pass per (i, j) pair because the pixel
        # tensors may differ in shape across samples.
        batch_neg_scores = None
        if exp_cfg.get("use_batch_negatives", True):
            sources = [m.get("source", "unknown") for m in cl_metadata]
            neg_indices = build_in_batch_negative_indices(batch_size, sources)

            batch_neg_scores = self._compute_cross_batch_negatives_generative(
                model, inputs, neg_indices,
                entity_masks=entity_masks, anchor_weights=anchor_weights,
            )

        # ── Compute grounding negative scores ──
        grounding_neg_scores = None
        if (
            exp_cfg.get("use_blackened", False)
            or exp_cfg.get("use_gaussian", False)
        ):
            grounding_neg_scores = self._compute_grounding_negatives_generative(
                model, inputs, cl_metadata,
                entity_masks=entity_masks, anchor_weights=anchor_weights,
            )

        # ── Compute temporal negative scores ──
        temporal_neg_scores = None
        if (
            exp_cfg.get("use_temporal_short", False)
            or exp_cfg.get("use_temporal_long", False)
        ):
            temporal_neg_scores = self._compute_temporal_negatives_generative(
                model, inputs, cl_metadata,
                entity_masks=entity_masks, anchor_weights=anchor_weights,
            )

        # ── (T-04) Compute MCQ-distractor answer negatives ──
        # Same forward pass as positive — zero extra compute.
        # For each sample, score EVERY wrong MCQ choice (typically 7) by
        # substituting its tokens at the gold-anchor position and
        # computing weighted log-likelihood under the same logits.
        # Samples with fewer/no distractors are padded with the positive
        # score so they contribute zero gradient pressure.
        answer_neg_scores = None
        if exp_cfg.get("use_answer_negatives", False):
            tokenizer = self.processing_class.tokenizer
            # Determine the max distractor count across the batch so we
            # can build a uniform [B, K] tensor.
            per_sample_distractors: List[List[str]] = []
            for i in range(batch_size):
                d_list: List[str] = []
                if cl_metadata is not None and i < len(cl_metadata):
                    d_list = list(
                        cl_metadata[i].get("distractor_anchor_texts", []) or []
                    )
                per_sample_distractors.append(d_list)
            max_k = max((len(d) for d in per_sample_distractors), default=0)

            if max_k > 0:
                neg_rows = []
                for i in range(batch_size):
                    gold_text = ""
                    if cl_metadata is not None and i < len(cl_metadata):
                        gold_text = cl_metadata[i].get("gold_anchor_text", "") or ""
                    d_list = per_sample_distractors[i]
                    row_scores = []
                    for k in range(max_k):
                        if k < len(d_list) and gold_text:
                            score = compute_distractor_negative_score(
                                positive_outputs.logits[i],
                                inputs["labels"][i],
                                gold_anchor_text=gold_text,
                                distractor_anchor_text=d_list[k],
                                tokenizer=tokenizer,
                                anchor_weight=anchor_weight,
                                base_weight=1.0,
                                ignore_index=IGNORE_INDEX,
                            )
                        else:
                            # Pad with positive score → contributes zero
                            # gradient signal (margin = 0 in InfoNCE).
                            score = positive_scores[i].detach().clone()
                        row_scores.append(score)
                    neg_rows.append(torch.stack(row_scores))
                answer_neg_scores = torch.stack(neg_rows)  # [B, K]

        # ── Combine all grounding negatives ──
        all_grounding = []
        if grounding_neg_scores is not None:
            all_grounding.append(grounding_neg_scores)
        if temporal_neg_scores is not None:
            all_grounding.append(temporal_neg_scores)
        if answer_neg_scores is not None:
            all_grounding.append(answer_neg_scores)

        combined_grounding = None
        if all_grounding:
            combined_grounding = torch.cat(all_grounding, dim=1)  # [B, N_total]

        # ── Compute InfoNCE loss ──
        if batch_neg_scores is None:
            # No batch negatives — use empty tensor so it doesn't pollute
            # the denominator. A zero score would act as a "perfect" negative
            # since actual scores are negative log-likelihoods.
            batch_neg_scores = torch.empty(
                batch_size, 0, device=device, dtype=positive_scores.dtype
            )

        cl_loss = compute_infonce_loss(
            positive_scores=positive_scores,
            batch_negative_scores=batch_neg_scores,
            grounding_negative_scores=combined_grounding,
            temperature=temperature,
            alpha=alpha,
        )

        return cl_loss

    # ───────────────────────────────────────────────────────────────
    # Helper: split flat pixel_values_videos into per-sample tensors
    # ───────────────────────────────────────────────────────────────

    @staticmethod
    def _split_pixels_per_sample(inputs: Dict[str, torch.Tensor], batch_size: int):
        """
        Split the flat-concatenated pixel_values_videos (and associated
        video_grid_thw / second_per_grid_ts) into per-sample lists.

        Qwen3-VL batches concatenate all samples' video patches into a
        single tensor of shape [N_total_patches, C].  video_grid_thw has
        one row [T, H, W] per video segment.  With 1 video per sample
        (our case), there are exactly `batch_size` rows and we can split
        by computing each row's patch count = T * H * W.

        Returns:
            per_sample_pixels:  list of B tensors, each [N_i, C]
            per_sample_grid:    list of B tensors, each [1, 3]
            per_sample_ts:      list of B lists  (or None)
        """
        pixel_key = (
            "pixel_values_videos" if "pixel_values_videos" in inputs
            else "pixel_values" if "pixel_values" in inputs
            else None
        )
        grid_key = (
            "video_grid_thw" if "video_grid_thw" in inputs
            else "image_grid_thw" if "image_grid_thw" in inputs
            else None
        )

        if pixel_key is None or grid_key is None:
            return None, None, None

        grid_thw = inputs[grid_key]               # [num_segments, 3]
        pixels = inputs[pixel_key]                 # [N_total, C]
        num_segments = grid_thw.shape[0]

        # Compute patch counts per segment
        patch_counts = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).tolist()

        # Assume 1 video segment per sample (holds for our datasets)
        # If there are more segments than batch_size, fall back gracefully
        if num_segments != batch_size:
            return None, None, None

        per_sample_pixels = list(pixels.split([int(c) for c in patch_counts], dim=0))
        per_sample_grid = [grid_thw[i : i + 1] for i in range(batch_size)]

        # Split second_per_grid_ts (list, one entry per segment)
        per_sample_ts = None
        if "second_per_grid_ts" in inputs:
            ts = inputs["second_per_grid_ts"]
            if isinstance(ts, list) and len(ts) == num_segments:
                per_sample_ts = [[ts[i]] for i in range(batch_size)]

        return per_sample_pixels, per_sample_grid, per_sample_ts

    # ───────────────────────────────────────────────────────────────
    # Proper cross-batch negatives for generative mode
    # ───────────────────────────────────────────────────────────────

    def _compute_cross_batch_negatives_generative(
        self,
        model,
        inputs: Dict[str, torch.Tensor],
        neg_indices: List[List[int]],
        entity_masks: Optional[List[torch.Tensor]] = None,
        anchor_weights: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        For each sample i, compute s(V_j, T_i) for every j in neg_indices[i].

        This is the CORRECT batch negative: "log-likelihood of sample i's
        answer tokens when the model sees sample j's video instead of
        sample i's video".  Requires one forward pass per (i, j) pair.

        Because input_ids contains a fixed number of <video> placeholder
        tokens determined at collation time, we can only swap pixels
        between samples whose video_grid_thw (and thus pixel count)
        match exactly.  When they don't match we fall back to blackened
        frames — the same safe fallback the temporal negatives use.

        With per_device_batch_size=2 this means at most 2 extra forward
        passes (under torch.no_grad()).
        """
        device = inputs["input_ids"].device
        batch_size = inputs["input_ids"].shape[0]

        pixel_key = (
            "pixel_values_videos" if "pixel_values_videos" in inputs
            else "pixel_values" if "pixel_values" in inputs
            else None
        )
        grid_key = (
            "video_grid_thw" if "video_grid_thw" in inputs
            else "image_grid_thw" if "image_grid_thw" in inputs
            else None
        )

        if pixel_key is None or grid_key is None:
            return torch.empty(batch_size, 0, device=device)

        # Split pixels per sample so we can swap them
        per_pixels, per_grid, per_ts = self._split_pixels_per_sample(
            inputs, batch_size
        )

        if per_pixels is None:
            return torch.empty(batch_size, 0, device=device)

        neg_score_lists = []  # list of B lists

        for i in range(batch_size):
            scores_i = []
            for j in neg_indices[i]:
                # Check if pixel shapes are compatible (same grid → same
                # number of video tokens in input_ids)
                if per_pixels[j].shape == per_pixels[i].shape:
                    # Compatible — build cross input: i's text + j's pixels
                    cross_inputs = {}
                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape[0] > i:
                            cross_inputs[k] = v[i : i + 1]
                        else:
                            cross_inputs[k] = v
                    cross_inputs[pixel_key] = per_pixels[j]
                    cross_inputs[grid_key] = per_grid[j]
                    if per_ts is not None:
                        cross_inputs["second_per_grid_ts"] = per_ts[j]
                else:
                    # Incompatible shapes — use blackened frames as fallback
                    # (same strategy as temporal negatives on shape mismatch)
                    cross_inputs = {}
                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape[0] > i:
                            cross_inputs[k] = v[i : i + 1]
                        else:
                            cross_inputs[k] = v
                    cross_inputs[pixel_key] = blacken_pixel_values(per_pixels[i])

                _ctx = contextlib.nullcontext() if self._grad_through_negs else torch.no_grad()
                with _ctx:
                    cross_out = model(**cross_inputs)

                if anchor_weights is not None:
                    score = compute_generation_log_likelihood_weighted(
                        cross_out.logits[0],
                        inputs["labels"][i],
                        anchor_weights[i],
                        ignore_index=IGNORE_INDEX,
                    )
                elif entity_masks is not None:
                    score = compute_generation_log_likelihood_masked(
                        cross_out.logits[0],
                        inputs["labels"][i],
                        entity_masks[i],
                        ignore_index=IGNORE_INDEX,
                    )
                else:
                    score = compute_generation_log_likelihood(
                        cross_out.logits[0],
                        inputs["labels"][i],  # sample i's answer tokens
                        ignore_index=IGNORE_INDEX,
                    )
                scores_i.append(score)

            if scores_i:
                neg_score_lists.append(torch.stack(scores_i))
            else:
                neg_score_lists.append(
                    torch.empty(0, device=device, dtype=torch.float)
                )

        # Pad to uniform length and stack → [B, max_neg]
        max_neg = max((t.shape[0] for t in neg_score_lists), default=0)
        if max_neg == 0:
            return torch.empty(batch_size, 0, device=device, dtype=torch.float)
        batch_neg_scores = torch.stack([
            F.pad(t, (0, max_neg - t.shape[0]), value=-1e9)
            for t in neg_score_lists
        ])
        return batch_neg_scores

    def _compute_grounding_negatives_generative(
        self,
        model,
        inputs: Dict[str, torch.Tensor],
        cl_metadata: List[dict],
        entity_masks: Optional[List[torch.Tensor]] = None,
        anchor_weights: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Stage: Generate blackened/gaussian negatives and compute their
        log-likelihood scores.
        
        For each eligible sample, we create a corrupted version of the
        pixel values (all-black or Gaussian noise), run a forward pass,
        and compute the answer log-likelihood under the corrupted video.
        
        Memory optimization: we do ONE batched forward pass with ALL
        corrupted pixel values at once.
        """
        device = inputs["input_ids"].device
        batch_size = inputs["input_ids"].shape[0]
        exp_cfg = self.experiment_config

        # Determine which pixel key to corrupt
        pixel_key = None
        if "pixel_values_videos" in inputs:
            pixel_key = "pixel_values_videos"
        elif "pixel_values" in inputs:
            pixel_key = "pixel_values"

        if pixel_key is None:
            return torch.empty(batch_size, 0, device=device)

        original_pixels = inputs[pixel_key]

        neg_scores_list = []

        # Blackened frames (V-02)
        _ctx_fn = lambda: contextlib.nullcontext() if self._grad_through_negs else torch.no_grad()

        if exp_cfg.get("use_blackened", False):
            corrupted_inputs = {k: v for k, v in inputs.items()}
            corrupted_inputs[pixel_key] = blacken_pixel_values(original_pixels)

            with _ctx_fn():
                corrupted_outputs = model(**corrupted_inputs)

            scores = []
            for i in range(batch_size):
                if anchor_weights is not None:
                    score = compute_generation_log_likelihood_weighted(
                        corrupted_outputs.logits[i],
                        inputs["labels"][i],
                        anchor_weights[i],
                        ignore_index=IGNORE_INDEX,
                    )
                elif entity_masks is not None:
                    score = compute_generation_log_likelihood_masked(
                        corrupted_outputs.logits[i],
                        inputs["labels"][i],
                        entity_masks[i],
                        ignore_index=IGNORE_INDEX,
                    )
                else:
                    score = compute_generation_log_likelihood(
                        corrupted_outputs.logits[i],
                        inputs["labels"][i],
                        ignore_index=IGNORE_INDEX,
                    )
                scores.append(score)
            neg_scores_list.append(torch.stack(scores).unsqueeze(1))  # [B, 1]

        # Gaussian noise (V-03)
        if exp_cfg.get("use_gaussian", False):
            corrupted_inputs = {k: v for k, v in inputs.items()}
            corrupted_inputs[pixel_key] = gaussianize_pixel_values(original_pixels)

            with _ctx_fn():
                corrupted_outputs = model(**corrupted_inputs)

            scores = []
            for i in range(batch_size):
                if anchor_weights is not None:
                    score = compute_generation_log_likelihood_weighted(
                        corrupted_outputs.logits[i],
                        inputs["labels"][i],
                        anchor_weights[i],
                        ignore_index=IGNORE_INDEX,
                    )
                elif entity_masks is not None:
                    score = compute_generation_log_likelihood_masked(
                        corrupted_outputs.logits[i],
                        inputs["labels"][i],
                        entity_masks[i],
                        ignore_index=IGNORE_INDEX,
                    )
                else:
                    score = compute_generation_log_likelihood(
                        corrupted_outputs.logits[i],
                        inputs["labels"][i],
                        ignore_index=IGNORE_INDEX,
                    )
                scores.append(score)
            neg_scores_list.append(torch.stack(scores).unsqueeze(1))  # [B, 1]

        if neg_scores_list:
            return torch.cat(neg_scores_list, dim=1)  # [B, num_neg_types]
        return torch.empty(batch_size, 0, device=device)

    def _compute_temporal_negatives_generative(
        self,
        model,
        inputs: Dict[str, torch.Tensor],
        cl_metadata: List[dict],
        entity_masks: Optional[List[torch.Tensor]] = None,
        anchor_weights: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Stage: Temporal negative generation (V-04, V-05, T-02, T-03).
        
        For each eligible sample with timestamps, extracts N temporal clips
        from the SAME video (where N = num_temporal_clips from cl_args).
        Each clip is a non-overlapping shifted segment, giving richer
        same-video contrastive signal than a single shift.
        
        Edge cases:
          - If shift overlaps GT despite wrapping → use blackened frames
          - FineVideo (no timestamps) → skip
          - Video too short for N clips → return as many as possible + pad
        """
        device = inputs["input_ids"].device
        batch_size = inputs["input_ids"].shape[0]
        exp_cfg = self.experiment_config
        num_clips = getattr(self.cl_args, "num_temporal_clips", 1)

        # Determine shift amount
        if exp_cfg.get("use_temporal_short", False):
            shift_seconds = 30.0
            shift_mode = "short"
        elif exp_cfg.get("use_temporal_long", False):
            shift_seconds = 120.0
            shift_mode = "long"
        else:
            return torch.empty(batch_size, 0, device=device)

        pixel_key = "pixel_values_videos" if "pixel_values_videos" in inputs else "pixel_values"

        # For each sample, collect a list of temporal negative scores
        all_sample_scores = []  # list of lists

        for i in range(batch_size):
            meta = cl_metadata[i]
            sample_scores = []

            # Skip non-eligible samples
            if not meta.get("cl_eligible", False) or not meta.get("has_timestamps", False):
                sample_scores = []
                all_sample_scores.append(sample_scores)
                continue

            timestamps_sec = meta.get("timestamps_sec", [])
            duration_sec = meta.get("duration_sec", 0.0)
            full_video_path = meta.get("full_video_path", "")

            if not full_video_path or not os.path.exists(full_video_path):
                sample_scores.append(
                    self._blackened_score_single(
                        model, inputs, i, pixel_key,
                        entity_mask=entity_masks[i] if entity_masks else None,
                        anchor_weight=anchor_weights[i] if anchor_weights else None,
                    )
                )
                all_sample_scores.append(sample_scores)
                continue

            # Get multiple non-overlapping temporal shifts from the same video
            shifted_clips = compute_multiple_temporal_shifts(
                timestamps_sec, duration_sec, shift_seconds, shift_mode,
                num_clips=num_clips,
            )

            if not shifted_clips:
                sample_scores.append(
                    self._blackened_score_single(
                        model, inputs, i, pixel_key,
                        entity_mask=entity_masks[i] if entity_masks else None,
                        anchor_weight=anchor_weights[i] if anchor_weights else None,
                    )
                )
                all_sample_scores.append(sample_scores)
                continue

            for shifted in shifted_clips:
                try:
                    shifted_score = self._load_and_score_temporal_clip(
                        model, inputs, i, full_video_path, shifted, pixel_key,
                        entity_mask=entity_masks[i] if entity_masks else None,
                        anchor_weight=anchor_weights[i] if anchor_weights else None,
                    )
                    sample_scores.append(shifted_score)
                except Exception as e:
                    if self._cl_step % 100 == 0:
                        logger.warning(
                            f"Temporal clip loading failed for {full_video_path}: {e}. "
                            f"Using blackened fallback."
                        )
                    sample_scores.append(
                        self._blackened_score_single(
                            model, inputs, i, pixel_key,
                            entity_mask=entity_masks[i] if entity_masks else None,
                            anchor_weight=anchor_weights[i] if anchor_weights else None,
                        )
                    )

            all_sample_scores.append(sample_scores)

        # Pad to uniform width and stack → [B, max_clips]
        max_clips = max((len(s) for s in all_sample_scores), default=0)
        if max_clips == 0:
            return torch.empty(batch_size, 0, device=device, dtype=torch.float)
        padded = []
        for scores in all_sample_scores:
            if scores:
                t = torch.stack(scores)
            else:
                t = torch.empty(0, device=device, dtype=torch.float)
            if t.shape[0] < max_clips:
                t = F.pad(t, (0, max_clips - t.shape[0]), value=-1e9)
            padded.append(t)

        return torch.stack(padded)  # [B, max_clips]

    def _blackened_score_single(
        self,
        model,
        inputs: Dict[str, torch.Tensor],
        sample_idx: int,
        pixel_key: str,
        entity_mask: Optional[torch.Tensor] = None,
        anchor_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Fallback: compute log-likelihood with blackened pixels for a single sample.
        Used when temporal shifting fails.
        """
        # Create a single-sample batch with blackened pixels
        single_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor) and v.shape[0] > sample_idx:
                single_inputs[k] = v[sample_idx : sample_idx + 1]
            else:
                single_inputs[k] = v

        if pixel_key in single_inputs:
            single_inputs[pixel_key] = blacken_pixel_values(single_inputs[pixel_key])

        _ctx = contextlib.nullcontext() if self._grad_through_negs else torch.no_grad()
        with _ctx:
            outputs = model(**single_inputs)

        if anchor_weight is not None:
            score = compute_generation_log_likelihood_weighted(
                outputs.logits[0],
                single_inputs["labels"][0],
                anchor_weight,
                ignore_index=IGNORE_INDEX,
            )
        elif entity_mask is not None:
            score = compute_generation_log_likelihood_masked(
                outputs.logits[0],
                single_inputs["labels"][0],
                entity_mask,
                ignore_index=IGNORE_INDEX,
            )
        else:
            score = compute_generation_log_likelihood(
                outputs.logits[0],
                single_inputs["labels"][0],
                ignore_index=IGNORE_INDEX,
        )
        return score

    def _load_and_score_temporal_clip(
        self,
        model,
        inputs: Dict[str, torch.Tensor],
        sample_idx: int,
        full_video_path: str,
        shifted_timestamps: Tuple[float, float],
        pixel_key: str,
        entity_mask: Optional[torch.Tensor] = None,
        anchor_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Stage: Load a temporally-shifted clip from the full video using
        the same qwen_vl_utils pipeline as the dataset, and compute its
        log-likelihood score.
        
        Uses video_start/video_end in the content dict (supported natively
        by _read_video_decord) to extract only the shifted temporal segment.
        """
        from qwen_vl_utils import process_vision_info

        dataset = self.train_dataset
        shift_start, shift_end = shifted_timestamps

        # Build the same content dict that get_video_info uses, but with
        # video_start/video_end to extract only the shifted segment.
        content = {
            "type": "video",
            "video": full_video_path,
            "video_start": shift_start,
            "video_end": shift_end,
            "min_pixels": dataset.video_min_pixel,
            "max_pixels": dataset.video_max_pixel,
        }
        if dataset.video_total_pixels is not None:
            content["total_pixels"] = dataset.video_total_pixels
        if dataset.video_max_frames is not None:
            content["max_frames"] = dataset.video_max_frames
        if dataset.fps is not None:
            content["fps"] = dataset.fps
        elif dataset.nframes is not None:
            content["nframes"] = dataset.nframes

        messages = [{"role": "user", "content": [content]}]

        try:
            _, video_input, _ = process_vision_info(
                messages,
                return_video_kwargs=True,
                image_patch_size=dataset.image_patch_size,
                return_video_metadata=dataset.return_video_metadata,
            )

            if dataset.return_video_metadata:
                shifted_pixels = video_input[0][0]  # (data, metadata) tuple → data
            else:
                shifted_pixels = video_input[0]
        except Exception as e:
            raise RuntimeError(
                f"process_vision_info failed for {full_video_path} "
                f"[{shift_start:.1f}-{shift_end:.1f}s]: {e}"
            )

        # Replace pixels in a single-sample input dict
        single_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape[0] > sample_idx:
                single_inputs[k] = v[sample_idx : sample_idx + 1]
            else:
                single_inputs[k] = v

        # The shifted clip may have different dimensions — we need to handle
        # the grid_thw accordingly. For simplicity, if the shapes don't match,
        # we fall back to blackened.
        if pixel_key in single_inputs:
            orig_shape = single_inputs[pixel_key].shape
            device = single_inputs[pixel_key].device
            dtype = single_inputs[pixel_key].dtype

            shifted_pixels_tensor = shifted_pixels.to(device=device, dtype=dtype)

            if shifted_pixels_tensor.shape == orig_shape:
                single_inputs[pixel_key] = shifted_pixels_tensor
            else:
                # Shape mismatch — fall back to blackened (safe fallback)
                single_inputs[pixel_key] = blacken_pixel_values(
                    single_inputs[pixel_key]
                )

        _ctx = contextlib.nullcontext() if self._grad_through_negs else torch.no_grad()
        with _ctx:
            outputs = model(**single_inputs)

        if anchor_weight is not None:
            score = compute_generation_log_likelihood_weighted(
                outputs.logits[0],
                single_inputs["labels"][0],
                anchor_weight,
                ignore_index=IGNORE_INDEX,
            )
        elif entity_mask is not None:
            score = compute_generation_log_likelihood_masked(
                outputs.logits[0],
                single_inputs["labels"][0],
                entity_mask,
                ignore_index=IGNORE_INDEX,
            )
        else:
            score = compute_generation_log_likelihood(
                outputs.logits[0],
                single_inputs["labels"][0],
                ignore_index=IGNORE_INDEX,
            )
        return score

    # ═══════════════════════════════════════════════════════════════
    # Stage: Vector contrastive mode (Amazon paper style)
    # ═══════════════════════════════════════════════════════════════

    def _compute_vector_contrastive_loss(
        self,
        model,
        inputs: Dict[str, torch.Tensor],
        positive_outputs,
        cl_metadata: List[dict],
        cached_hidden_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Vector contrastive mode (following Amazon paper):
        
        1. Extract last hidden states from the model
        2. Extract the [EOS] hidden state per sample
        3. Project through ContrastiveProjectionHead (the only learnable CL component)
        4. Compute cosine similarity as s(V, T)
        5. Build negatives (blackened EOS embeddings, batch EOS embeddings)
        6. Compute InfoNCE loss
        
        NOTE: Qwen3VLForConditionalGeneration.forward() does NOT propagate
        hidden_states in its output (even with output_hidden_states=True).
        The inner Qwen3VLModel returns last_hidden_state as outputs[0].
        We capture this via a forward hook registered in compute_loss()
        BEFORE the main forward pass, so all ranks participate.
        
        The captured hidden states are detached (no grad from the backbone).
        Only the projection_head parameters receive gradients from CL loss.
        """
        device = inputs["input_ids"].device
        batch_size = inputs["input_ids"].shape[0]
        exp_cfg = self.experiment_config
        temperature = self.cl_args.contrastive_temperature
        alpha = exp_cfg["default_alpha"]

        if self.projection_head is None:
            logger.warning(
                "Vector contrastive mode requires a projection_head. "
                "Falling back to zero CL loss."
            )
            return torch.tensor(0.0, device=device)

        # ── Resolve EOS token id ──
        eos_token_id = self.processing_class.tokenizer.eos_token_id
        if eos_token_id is None:
            eos_token_id = self.processing_class.tokenizer.convert_tokens_to_ids(
                "<|im_end|>"
            )

        # ── Step 1: Get last hidden states from positive forward pass ──
        # Use cached hidden state from the main SFT forward (captured via
        # hook in compute_loss). This avoids an extra model(**inputs) call
        # that would cause NCCL deadlocks when ranks differ in CL eligibility.
        if cached_hidden_state is not None:
            last_hidden = cached_hidden_state.detach()  # [B, seq_len, D]
        else:
            # Fallback: run a separate forward (only safe in single-GPU)
            last_hidden = self._get_last_hidden_state(model, inputs)

        # ── Step 2: Extract EOS hidden states ──
        eos_hidden = extract_eos_hidden_states(
            last_hidden, inputs["input_ids"], eos_token_id
        )  # [B, D]

        # ── Step 3: Project through head (WITH gradient for proj_head params) ──
        positive_embeddings = self.projection_head(eos_hidden)  # [B, proj_dim]

        # ── Step 4: Positive self-similarity = 1.0 (L2-normalized) ──
        positive_scores = torch.sum(
            positive_embeddings * positive_embeddings, dim=-1
        )  # [B] — all ~1.0

        # ── Step 5: Batch negatives via cross-similarity matrix ──
        sim_matrix = torch.mm(
            positive_embeddings, positive_embeddings.t()
        )  # [B, B]

        sources = [m.get("source", "unknown") for m in cl_metadata]
        neg_indices = build_in_batch_negative_indices(batch_size, sources)

        batch_neg_scores = []
        for i in range(batch_size):
            idx = neg_indices[i]
            if idx:
                batch_neg_scores.append(sim_matrix[i, idx])
            else:
                batch_neg_scores.append(
                    torch.empty(0, device=device, dtype=sim_matrix.dtype)
                )
        max_neg = max((t.shape[0] for t in batch_neg_scores), default=0)
        if max_neg == 0:
            batch_neg_scores = torch.empty(
                batch_size, 0, device=device, dtype=sim_matrix.dtype
            )
        else:
            batch_neg_scores = torch.stack([
                F.pad(t, (0, max_neg - t.shape[0]), value=-1e9)
                for t in batch_neg_scores
            ])  # [B, max_neg]

        # ── Step 6: Grounding negatives (blackened/gaussian embeddings) ──
        grounding_neg_scores = None
        pixel_key = (
            "pixel_values_videos"
            if "pixel_values_videos" in inputs
            else "pixel_values" if "pixel_values" in inputs else None
        )

        if pixel_key and (
            exp_cfg.get("use_blackened", False) or exp_cfg.get("use_gaussian", False)
        ):
            grounding_neg_list = []

            if exp_cfg.get("use_blackened", False):
                corrupted = {k: v for k, v in inputs.items()}
                corrupted[pixel_key] = blacken_pixel_values(inputs[pixel_key])
                _ctx = contextlib.nullcontext() if self._grad_through_negs else torch.no_grad()
                with _ctx:
                    corr_hidden = self._get_last_hidden_state(model, corrupted)
                corr_eos = extract_eos_hidden_states(
                    corr_hidden, inputs["input_ids"], eos_token_id
                )
                corr_emb = self.projection_head(corr_eos)  # [B, proj_dim]
                black_scores = torch.sum(
                    positive_embeddings * corr_emb, dim=-1
                ).unsqueeze(1)  # [B, 1]
                grounding_neg_list.append(black_scores)

            if exp_cfg.get("use_gaussian", False):
                corrupted = {k: v for k, v in inputs.items()}
                corrupted[pixel_key] = gaussianize_pixel_values(inputs[pixel_key])
                _ctx = contextlib.nullcontext() if self._grad_through_negs else torch.no_grad()
                with _ctx:
                    corr_hidden = self._get_last_hidden_state(model, corrupted)
                corr_eos = extract_eos_hidden_states(
                    corr_hidden, inputs["input_ids"], eos_token_id
                )
                corr_emb = self.projection_head(corr_eos)
                gauss_scores = torch.sum(
                    positive_embeddings * corr_emb, dim=-1
                ).unsqueeze(1)
                grounding_neg_list.append(gauss_scores)

            if grounding_neg_list:
                grounding_neg_scores = torch.cat(grounding_neg_list, dim=1)

        # ── Step 7: InfoNCE loss ──
        cl_loss = compute_infonce_loss(
            positive_scores=positive_scores,
            batch_negative_scores=batch_neg_scores,
            grounding_negative_scores=grounding_neg_scores,
            temperature=temperature,
            alpha=alpha,
        )

        return cl_loss

    def _find_inner_model(self, model):
        """
        Locate the inner Qwen3VLModel from the (possibly wrapped) model.
        Returns None if not found.
        
        Model hierarchy:
          DeepSpeedEngine → PeftModel → LoraModel → Qwen3VLForCond → Qwen3VLModel
        """
        unwrapped = self.accelerator.unwrap_model(model)
        base = getattr(unwrapped, "base_model", unwrapped)
        base = getattr(base, "model", base)          # LoraModel.model
        inner = getattr(base, "model", None)          # Qwen3VLForCond.model
        return inner

    def _get_last_hidden_state(
        self,
        model,
        inputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Extract the last-layer hidden states by running the UNWRAPPED model
        (not the DeepSpeed engine) under torch.no_grad().

        Using the unwrapped model avoids triggering NCCL collective
        operations that would deadlock when only some ranks call this.
        ZeRO-1 doesn't partition model params, so unwrapped forward is safe.

        A forward hook on the inner Qwen3VLModel captures last_hidden_state,
        which Qwen3VLForConditionalGeneration.forward() does NOT expose.
        """
        # ── Locate the unwrapped model and inner Qwen3VLModel ──
        unwrapped = self.accelerator.unwrap_model(model)
        inner_model = self._find_inner_model(model)
        if inner_model is None:
            raise RuntimeError(
                "Cannot find inner Qwen3VLModel; model hierarchy: "
                f"{type(unwrapped).__name__}"
            )

        # ── Register a one-shot forward hook to capture last_hidden_state ──
        captured = {}

        def _hook(module, input, output):
            if hasattr(output, "last_hidden_state"):
                captured["lhs"] = output.last_hidden_state
            else:
                captured["lhs"] = output[0]

        handle = inner_model.register_forward_hook(_hook)
        try:
            _ctx = contextlib.nullcontext() if self._grad_through_negs else torch.no_grad()
            with _ctx:
                # Run through the UNWRAPPED model (no DeepSpeed NCCL)
                unwrapped(**inputs)
        finally:
            handle.remove()

        if "lhs" not in captured:
            raise RuntimeError(
                "Forward hook on inner Qwen3VLModel did not fire. "
                f"inner_model type = {type(inner_model).__name__}"
            )

        return captured["lhs"]  # [B, seq_len, D]

    # ═══════════════════════════════════════════════════════════════
    # Stage: Logging
    # ═══════════════════════════════════════════════════════════════

    def _log_cl_metrics(
        self,
        sft_loss: torch.Tensor,
        cl_loss: torch.Tensor,
        total_loss: torch.Tensor,
        eligible_fraction: float,
    ):
        """Log contrastive learning metrics to the configured logger."""
        metrics = {
            "cl/sft_loss": sft_loss.item(),
            "cl/contrastive_loss": cl_loss.item(),
            "cl/total_loss": total_loss.item(),
            "cl/eligible_fraction": eligible_fraction,
            "cl/lambda": self.cl_args.contrastive_weight,
            "cl/alpha": self.experiment_config["default_alpha"],
            "cl/temperature": self.cl_args.contrastive_temperature,
        }
        self.log(metrics)

    # ═══════════════════════════════════════════════════════════════
    # Stage: Evaluation (same generation-based eval as vanilla SFT)
    # ═══════════════════════════════════════════════════════════════

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Override to handle cl_metadata in inputs."""
        # Pop cl_metadata before sending to model
        inputs.pop("cl_metadata", None)

        labels = inputs.get("labels") if "labels" in inputs else None

        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs.loss if hasattr(outputs, "loss") else None
            logits = outputs.logits if hasattr(outputs, "logits") else None

        if prediction_loss_only:
            return (loss, None, None)
        return (loss, logits, labels)

    def _extract_prompt_and_reference(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        tokenizer,
    ) -> tuple:
        """Extract prompt and reference answer from input_ids and labels."""
        label_mask = labels != IGNORE_INDEX
        if label_mask.any():
            answer_start_idx = label_mask.nonzero(as_tuple=True)[0][0].item()
        else:
            answer_start_idx = len(input_ids)
        prompt_ids = input_ids[:answer_start_idx]
        answer_ids = labels[label_mask]
        reference_text = tokenizer.decode(answer_ids, skip_special_tokens=True)
        return prompt_ids, reference_text

    def _prepare_generation_inputs(
        self,
        batch_prompt_ids: List[torch.Tensor],
        original_inputs: Dict[str, torch.Tensor],
        tokenizer,
        device,
    ) -> Dict[str, torch.Tensor]:
        """Prepare inputs for generation by padding prompts."""
        batch_size = len(batch_prompt_ids)
        max_prompt_len = max(p.shape[0] for p in batch_prompt_ids)

        padded_prompts = torch.full(
            (batch_size, max_prompt_len),
            tokenizer.pad_token_id,
            dtype=batch_prompt_ids[0].dtype,
            device=device,
        )
        attention_masks = torch.zeros(
            (batch_size, max_prompt_len), dtype=torch.long, device=device
        )

        for i, prompt in enumerate(batch_prompt_ids):
            prompt_len = len(prompt)
            padded_prompts[i, :prompt_len] = prompt
            attention_masks[i, :prompt_len] = 1

        gen_inputs = {
            "input_ids": padded_prompts,
            "attention_mask": attention_masks,
        }

        for key in [
            "pixel_values",
            "image_grid_thw",
            "pixel_values_videos",
            "video_grid_thw",
            "second_per_grid_ts",
        ]:
            if key in original_inputs:
                gen_inputs[key] = original_inputs[key]

        return gen_inputs

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Override evaluation_loop to support generation-based evaluation.
        Same as vanilla SFT trainer.
        """
        args = self.args
        prediction_loss_only = (
            prediction_loss_only
            if prediction_loss_only is not None
            else args.prediction_loss_only
        )

        if prediction_loss_only or self.compute_metrics is None:
            return super().evaluation_loop(
                dataloader,
                description,
                prediction_loss_only,
                ignore_keys,
                metric_key_prefix,
            )

        logger.info(f"\n***** Running {description} (Generation Mode) *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        logger.info(f"  Batch size = {self.args.eval_batch_size}")

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)
        model.eval()

        tokenizer = self.processing_class.tokenizer

        generation_config = GenerationConfig(
            do_sample=False,
            max_new_tokens=getattr(args, "generation_max_new_tokens", 512),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        unwrapped_model = self.accelerator.unwrap_model(model)

        all_predictions = []
        all_references = []
        all_losses = []

        for step, inputs in enumerate(dataloader):
            # Pop cl_metadata before model forward
            inputs.pop("cl_metadata", None)
            inputs = self._prepare_inputs(inputs)

            batch_input_ids = inputs["input_ids"]
            batch_labels = inputs["labels"]
            batch_size = batch_input_ids.shape[0]

            with torch.no_grad():
                outputs = model(**inputs)
                if hasattr(outputs, "loss") and outputs.loss is not None:
                    loss = outputs.loss.detach()
                    loss = self.accelerator.gather(loss.repeat(batch_size))
                    all_losses.append(loss.cpu())

            batch_prompt_ids = []
            batch_references = []
            for i in range(batch_size):
                prompt_ids, reference_text = self._extract_prompt_and_reference(
                    batch_input_ids[i], batch_labels[i], tokenizer
                )
                batch_prompt_ids.append(prompt_ids)
                batch_references.append(reference_text)

            gen_inputs = self._prepare_generation_inputs(
                batch_prompt_ids, inputs, tokenizer, batch_input_ids.device
            )

            with torch.no_grad():
                generated_ids = unwrapped_model.generate(
                    **gen_inputs, generation_config=generation_config
                )

            for i in range(batch_size):
                prompt_len = len(batch_prompt_ids[i])
                new_tokens = generated_ids[i][prompt_len:]
                pred_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                all_predictions.append(pred_text)

            all_references.extend(batch_references)

            if step % 10 == 0:
                logger.info(f"  Eval step {step}/{len(dataloader)}")

        if self.args.world_size > 1:
            all_predictions = self._gather_predictions(all_predictions)
            all_references = self._gather_predictions(all_references)

        eval_prediction = GenerativeEvalPrediction(
            predictions=all_predictions, references=all_references
        )

        metrics = self.compute_metrics(eval_prediction)

        if all_losses:
            avg_loss = torch.cat(all_losses).mean().item()
            metrics[f"{metric_key_prefix}_loss"] = avg_loss

        metrics = {
            f"{metric_key_prefix}_{k}" if not k.startswith(metric_key_prefix) else k: v
            for k, v in metrics.items()
        }

        self.log(metrics)

        return EvalLoopOutput(
            predictions=all_predictions,
            label_ids=all_references,
            metrics=metrics,
            num_samples=len(all_predictions),
        )

    def _gather_predictions(self, predictions: List[str]) -> List[str]:
        """Gather string predictions across all processes."""
        import torch.distributed as dist

        if not dist.is_initialized():
            return predictions

        world_size = dist.get_world_size()
        gathered = [None] * world_size
        dist.all_gather_object(gathered, predictions)
        all_predictions = []
        for preds in gathered:
            all_predictions.extend(preds)
        return all_predictions
