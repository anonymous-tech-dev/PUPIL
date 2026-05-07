"""
==============================================================================
Contrastive Learning Utilities
==============================================================================
Helper functions for generating negative samples used in contrastive SFT.
These are called at collation/training time — NO videos are saved to disk.

Stage: Negative Sample Generation (runtime, in-memory)
==============================================================================
"""

import math
import random
import re
from typing import Dict, List, Optional, Tuple, Union

import torch
import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# Stage 1: Pixel-level negative generation (operates on processed tensors)
# ═══════════════════════════════════════════════════════════════════════

def blacken_pixel_values(pixel_values: torch.Tensor) -> torch.Tensor:
    """
    Replace all pixel values with zeros (black frames).
    Used for V-02 (blackened frames grounding penalty).
    
    Args:
        pixel_values: Tensor of shape [N, C] or [N, C, H, W] — the processed
                      pixel values from the Qwen processor.
    Returns:
        Zeroed tensor of the same shape and dtype.
    """
    return torch.zeros_like(pixel_values)


def gaussianize_pixel_values(
    pixel_values: torch.Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Replace all pixel values with Gaussian noise.
    Used for V-03 (Gaussian noise grounding test).
    
    Args:
        pixel_values: Tensor of shape [N, C] or [N, C, H, W].
        mean: Mean of the Gaussian distribution.
        std:  Standard deviation of the Gaussian distribution.
        seed: Optional seed for reproducibility.
    Returns:
        Gaussian noise tensor of the same shape and dtype.
    """
    if seed is not None:
        gen = torch.Generator(device=pixel_values.device)
        gen.manual_seed(seed)
        noise = torch.randn(
            pixel_values.shape,
            generator=gen,
            dtype=pixel_values.dtype,
            device=pixel_values.device,
        )
    else:
        noise = torch.randn_like(pixel_values)
    return noise * std + mean


# ═══════════════════════════════════════════════════════════════════════
# Stage 2: Timestamp parsing & temporal shifting
# ═══════════════════════════════════════════════════════════════════════

def parse_timestamp_str(ts_str: str) -> float:
    """
    Parse a timestamp string like "00:03:57" into seconds.
    Supports HH:MM:SS and MM:SS formats.
    """
    parts = ts_str.strip().split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    else:
        return float(parts[0])


def get_timestamp_bounds(timestamps_sec: List[List[float]]) -> Tuple[float, float]:
    """
    Given a list of timestamp segments like [[start1, end1], [start2, end2], ...],
    return the overall lower bound (min start) and upper bound (max end).
    
    This is used for multi-segment timestamps where we need the full span.
    For the contrastive temporal shift, we pick the lower clip boundary from
    the earliest segment and the upper clip boundary from the latest segment.
    """
    if not timestamps_sec:
        return 0.0, 0.0
    all_starts = [seg[0] for seg in timestamps_sec]
    all_ends = [seg[1] for seg in timestamps_sec]
    return min(all_starts), max(all_ends)


def intervals_overlap(
    a_start: float, a_end: float,
    b_segments: List[List[float]],
) -> bool:
    """
    Check if interval [a_start, a_end] overlaps with ANY segment in b_segments.
    Used to ensure that a temporal-shifted clip never overlaps with the
    ground-truth timestamp.
    """
    for seg in b_segments:
        seg_start, seg_end = seg[0], seg[1]
        if a_start < seg_end and a_end > seg_start:
            return True
    return False


def compute_temporal_shift(
    gt_timestamps_sec: List[List[float]],
    duration_sec: float,
    shift_seconds: float,
    mode: str = "short",
) -> Optional[Tuple[float, float]]:
    """
    Compute a temporally-shifted clip from the same video that does NOT
    overlap with the ground truth timestamps.
    
    Used for V-04 (short temporal shift, ±30s) and V-05 (long temporal
    shift, >120s).
    
    Args:
        gt_timestamps_sec: List of [start, end] pairs (ground truth segments).
        duration_sec:      Total video duration in seconds.
        shift_seconds:     How far to shift (30 for short, 120+ for long).
        mode:              "short" or "long" — determines shift direction logic.
    
    Returns:
        (new_start, new_end) tuple, or None if no valid shift exists
        (in which case the caller should fall back to blackened frames).
    
    Edge case handling:
        - If the clip is in the first minute and we want to go 2 minutes back,
          we wrap around to the end of the video.
        - If the shifted clip STILL overlaps with GT despite wrapping,
          we return None → caller uses blackened frames.
    """
    if not gt_timestamps_sec or duration_sec <= 0:
        return None

    gt_start, gt_end = get_timestamp_bounds(gt_timestamps_sec)
    clip_duration = gt_end - gt_start

    if clip_duration <= 0:
        clip_duration = min(10.0, duration_sec * 0.05)  # fallback: 10s or 5% of video

    # Try multiple shift candidates (forward, backward, wrapped)
    candidates = []

    # Forward shift
    fwd_start = gt_end + shift_seconds
    fwd_end = fwd_start + clip_duration
    candidates.append((fwd_start, fwd_end))

    # Backward shift
    bwd_end = gt_start - shift_seconds
    bwd_start = bwd_end - clip_duration
    candidates.append((bwd_start, bwd_end))

    # Wrapped forward (when forward goes past video end)
    if fwd_end > duration_sec:
        wrap_start = (fwd_start % duration_sec) if duration_sec > 0 else 0
        wrap_end = wrap_start + clip_duration
        candidates.append((wrap_start, wrap_end))

    # Wrapped backward (when backward goes before video start)
    if bwd_start < 0:
        wrap_end = duration_sec + bwd_end  # wrap to end of video
        wrap_start = wrap_end - clip_duration
        if wrap_start < 0:
            wrap_start = 0
        candidates.append((wrap_start, wrap_end))

    # For "long" mode, also try midpoints far from GT
    if mode == "long":
        # Try the point furthest from the GT center
        gt_center = (gt_start + gt_end) / 2
        opposite_center = (gt_center + duration_sec / 2) % duration_sec
        opp_start = opposite_center - clip_duration / 2
        opp_end = opposite_center + clip_duration / 2
        if opp_start < 0:
            opp_start = 0
            opp_end = clip_duration
        if opp_end > duration_sec:
            opp_end = duration_sec
            opp_start = max(0, opp_end - clip_duration)
        candidates.append((opp_start, opp_end))

    # Filter: must be within [0, duration] and must NOT overlap with GT
    random.shuffle(candidates)  # randomize which valid candidate we pick
    for cand_start, cand_end in candidates:
        # Clamp to valid range
        cand_start = max(0, cand_start)
        cand_end = min(duration_sec, cand_end)
        if cand_end - cand_start < 1.0:
            continue  # too short
        if not intervals_overlap(cand_start, cand_end, gt_timestamps_sec):
            return (cand_start, cand_end)

    # No valid candidate found — caller should use blackened frames as fallback
    return None


def compute_multiple_temporal_shifts(
    gt_timestamps_sec: List[List[float]],
    duration_sec: float,
    shift_seconds: float,
    mode: str = "short",
    num_clips: int = 1,
) -> List[Tuple[float, float]]:
    """
    Generate up to `num_clips` non-overlapping temporal shifts from the same
    video. Each clip avoids the GT timestamps AND all previously selected clips.

    This gives richer same-video contrastive signal: instead of 1 temporal
    negative, we get N temporal negatives that each show a different part of
    the video, all contrasted against the anchor (GT segment).

    Returns:
        List of (start, end) tuples (may be shorter than num_clips if the
        video is too short to find enough non-overlapping segments).
    """
    if num_clips <= 1:
        result = compute_temporal_shift(gt_timestamps_sec, duration_sec, shift_seconds, mode)
        return [result] if result is not None else []

    if not gt_timestamps_sec or duration_sec <= 0:
        return []

    gt_start, gt_end = get_timestamp_bounds(gt_timestamps_sec)
    clip_duration = gt_end - gt_start
    if clip_duration <= 0:
        clip_duration = min(10.0, duration_sec * 0.05)

    # Build list of forbidden intervals (GT + already-selected clips)
    forbidden = list(gt_timestamps_sec)
    results = []

    for _ in range(num_clips):
        # Generate candidates at various offsets, avoiding forbidden intervals
        candidates = []
        # Evenly spaced probes + random probes
        for offset in range(0, int(duration_sec), max(1, int(clip_duration * 1.5))):
            cand_start = float(offset)
            cand_end = cand_start + clip_duration
            if cand_end > duration_sec:
                continue
            if not intervals_overlap(cand_start, cand_end, forbidden):
                candidates.append((cand_start, cand_end))

        if not candidates:
            break  # Video too short for more clips

        random.shuffle(candidates)
        chosen = candidates[0]
        results.append(chosen)
        forbidden.append([chosen[0], chosen[1]])

    return results


# ═══════════════════════════════════════════════════════════════════════
# Stage 3: In-batch contrastive index management
# ═══════════════════════════════════════════════════════════════════════

def build_in_batch_negative_indices(
    batch_size: int,
    sources: Optional[List[str]] = None,
) -> List[List[int]]:
    """
    For in-batch contrastive learning: given a batch of size B, for each
    sample i, return the indices of all other samples that serve as
    batch negatives (V-01 / T-01).
    
    Memory-optimal: in a batch of 8, sample 0 is positive, samples 1-7
    are negatives, then sample 1 is positive, etc.
    
    If `sources` is provided, only samples from the SAME source are used
    as negatives (prevents cross-dataset noise).
    
    Args:
        batch_size: Number of samples in the batch.
        sources:    Optional list of source strings for each sample.
    Returns:
        List of lists, where result[i] = indices of negatives for sample i.
    """
    negatives = []
    for i in range(batch_size):
        if sources is not None:
            neg_idx = [
                j for j in range(batch_size)
                if j != i and sources[j] == sources[i]
            ]
        else:
            neg_idx = [j for j in range(batch_size) if j != i]
        negatives.append(neg_idx)
    return negatives


# ═══════════════════════════════════════════════════════════════════════
# Stage 4: Similarity scoring helpers
# ═══════════════════════════════════════════════════════════════════════

def cosine_similarity_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise cosine similarity between rows of a and b.
    
    Args:
        a: [N, D] tensor
        b: [M, D] tensor
    Returns:
        [N, M] similarity matrix
    """
    a_norm = torch.nn.functional.normalize(a, dim=-1)
    b_norm = torch.nn.functional.normalize(b, dim=-1)
    return torch.mm(a_norm, b_norm.t())


def compute_infonce_loss(
    positive_scores: torch.Tensor,
    batch_negative_scores: torch.Tensor,
    grounding_negative_scores: Optional[torch.Tensor] = None,
    temperature: float = 0.07,
    alpha: float = 5.0,
) -> torch.Tensor:
    """
    Compute the InfoNCE contrastive loss with optional grounding penalty.
    
    This implements the formula:
        L_grounding = -log( exp(s(V_i, T_i) / τ) /
            (Σ_j exp(s(V_j, T_i) / τ) + α * Σ_k exp(s(V_k, T_i) / τ)) )
    
    Uses the log-sum-exp trick for numerical stability: instead of computing
    exp() directly (which overflows/underflows with τ=0.07), we work in
    log-space throughout and use torch.logsumexp.
    
    Where:
        - positive_scores: s(V_i, T_i) for each sample in batch [B]
        - batch_negative_scores: s(V_j, T_i) for batch negatives [B, N_batch]
        - grounding_negative_scores: s(V_k, T_i) for corrupted negatives [B, N_ground]
        - temperature: τ
        - alpha: α (grounding penalty weight)
    
    Args:
        positive_scores:           [B] — similarity of positive pairs
        batch_negative_scores:     [B, N_batch] — similarities with batch negatives
        grounding_negative_scores: [B, N_ground] or None — similarities with
                                   corrupted/temporal negatives
        temperature: InfoNCE temperature
        alpha: Grounding penalty multiplier
    Returns:
        Scalar loss (mean over batch).
    """
    # Scaled scores (all in log-space: score / τ)
    pos_scaled = positive_scores / temperature  # [B]

    # All batch scores (positive + batch negatives) scaled
    all_batch_scaled = torch.cat([
        pos_scaled.unsqueeze(1),
        batch_negative_scores / temperature,
    ], dim=1)  # [B, 1 + N_batch]

    # Log-sum-exp of batch scores (numerically stable)
    log_denom_batch = torch.logsumexp(all_batch_scaled, dim=1)  # [B]

    # Grounding penalty: α * Σ_k exp(s_k / τ)
    # In log-space: log(α) + logsumexp(s_k / τ)
    if grounding_negative_scores is not None and grounding_negative_scores.numel() > 0:
        ground_scaled = grounding_negative_scores / temperature  # [B, N_ground]
        log_ground_sum = torch.logsumexp(ground_scaled, dim=1)  # [B]
        log_alpha = torch.tensor(alpha, device=pos_scaled.device, dtype=pos_scaled.dtype).log()
        log_denom_ground = log_alpha + log_ground_sum  # [B]
        
        # log(exp(a) + exp(b)) = logsumexp([a, b])
        log_denom = torch.logsumexp(
            torch.stack([log_denom_batch, log_denom_ground], dim=1), dim=1
        )  # [B]
    else:
        log_denom = log_denom_batch

    # InfoNCE loss = -log(exp(pos/τ) / denom) = -(pos/τ - log(denom))
    loss = -(pos_scaled - log_denom)  # [B]

    # Clamp to avoid extreme values from edge cases
    loss = loss.clamp(min=0.0, max=100.0)

    # Guard against NaN (can happen with degenerate batches)
    if torch.isnan(loss).any() or torch.isinf(loss).any():
        valid_mask = ~(torch.isnan(loss) | torch.isinf(loss))
        if valid_mask.any():
            return loss[valid_mask].mean()
        return torch.tensor(0.0, device=loss.device, dtype=loss.dtype)

    return loss.mean()


# ═══════════════════════════════════════════════════════════════════════
# Stage 5: Generative-mode scoring (log-likelihood)
# ═══════════════════════════════════════════════════════════════════════

def compute_generation_log_likelihood(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Compute the average per-token log-probability of the answer tokens.
    Used as the similarity score s(V, T) in generative contrastive mode.
    
    A higher (less negative) log-likelihood means the model assigns higher
    probability to generating the correct answer given the visual input.
    
    Args:
        logits: [seq_len, vocab_size] — model output logits
        labels: [seq_len] — token IDs with IGNORE_INDEX for non-answer tokens
    Returns:
        Scalar: mean log-probability of answer tokens.
    """
    # Shift logits and labels for next-token prediction
    shift_logits = logits[:-1]  # [seq_len-1, vocab]
    shift_labels = labels[1:]   # [seq_len-1]

    # Mask to answer tokens only
    mask = shift_labels != ignore_index
    if not mask.any():
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

    # Compute per-token log probabilities
    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(
        dim=-1, index=shift_labels[mask].unsqueeze(-1)
    ).squeeze(-1)  # [num_answer_tokens]

    # Return mean log-likelihood as similarity score
    return token_log_probs.mean()


# ═══════════════════════════════════════════════════════════════════════
# Stage 6: Vector-mode scoring (EOS hidden state projection)
# ═══════════════════════════════════════════════════════════════════════

class ContrastiveProjectionHead(torch.nn.Module):
    """
    Linear projection head for vector-mode contrastive learning.
    
    Following the Amazon paper "Aligning VLMs with Contrastive Learning":
    We extract the LLM's final hidden state at the [EOS] token from the
    template "[X] means: [EOS]", then project it through this linear layer
    to compute cosine similarity.
    
    The template forces the model to compress all meaning into one token,
    which is then used for contrastive comparison.
    """

    def __init__(self, hidden_size: int, projection_dim: int = 512):
        super().__init__()
        self.projection = torch.nn.Linear(hidden_size, projection_dim, bias=False)
        # Initialize with small values for stability
        torch.nn.init.xavier_uniform_(self.projection.weight)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_state: [B, D] — the hidden state at the EOS token
        Returns:
            [B, projection_dim] — projected and L2-normalized embedding
        """
        projected = self.projection(hidden_state)
        return torch.nn.functional.normalize(projected, dim=-1)


def extract_eos_hidden_states(
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    eos_token_id: int,
) -> torch.Tensor:
    """
    Extract the hidden state at the LAST [EOS] token position for each
    sequence in the batch.
    
    Following the Amazon paper: "[X] means: [EOS]" — we extract at the
    exact EOS position, not a random pool.
    
    Args:
        hidden_states: [B, seq_len, D] — last layer hidden states
        input_ids:     [B, seq_len] — input token IDs
        eos_token_id:  Token ID for EOS
    Returns:
        [B, D] — one hidden state vector per batch element
    """
    batch_size = input_ids.shape[0]
    device = hidden_states.device
    
    eos_positions = []
    for i in range(batch_size):
        eos_mask = (input_ids[i] == eos_token_id)
        if eos_mask.any():
            # Use the LAST EOS token position
            pos = eos_mask.nonzero(as_tuple=True)[0][-1].item()
        else:
            # Fallback: use the last non-padding token
            non_pad = (input_ids[i] != 0).nonzero(as_tuple=True)[0]
            pos = non_pad[-1].item() if non_pad.numel() > 0 else input_ids.shape[1] - 1
        eos_positions.append(pos)

    eos_positions = torch.tensor(eos_positions, device=device)
    # Gather: [B, D]
    eos_hidden = hidden_states[
        torch.arange(batch_size, device=device),
        eos_positions,
    ]
    return eos_hidden


# ═══════════════════════════════════════════════════════════════════════
# Stage 7: Experiment configuration mapping
# ═══════════════════════════════════════════════════════════════════════

# Maps experiment IDs to which negative strategies to use
EXPERIMENT_CONFIGS = {
    # Visual grounding experiments (V, Q, A) vs (~V, Q, A)
    "V-01": {
        "description": "Batch negatives only (baseline InfoNCE)",
        "use_batch_negatives": True,
        "use_blackened": False,
        "use_gaussian": False,
        "use_temporal_short": False,
        "use_temporal_long": False,
        "use_answer_negatives": False,
        "default_alpha": 1.0,
    },
    "V-02": {
        "description": "Blackened frames for grounding penalty",
        "use_batch_negatives": True,
        "use_blackened": True,
        "use_gaussian": False,
        "use_temporal_short": False,
        "use_temporal_long": False,
        "use_answer_negatives": False,
        "default_alpha": 5.0,
    },
    "V-03": {
        "description": "Gaussian noise frames",
        "use_batch_negatives": True,
        "use_blackened": False,
        "use_gaussian": True,
        "use_temporal_short": False,
        "use_temporal_long": False,
        "use_answer_negatives": False,
        "default_alpha": 5.0,
    },
    "V-04": {
        "description": "Temporal shift (short, ±30s) from same video",
        "use_batch_negatives": False,
        "use_blackened": False,
        "use_gaussian": False,
        "use_temporal_short": True,
        "use_temporal_long": False,
        "use_answer_negatives": False,
        "default_alpha": 1.0,
    },
    "V-05": {
        "description": "Temporal shift (long, >2min) from same video",
        "use_batch_negatives": True,
        "use_blackened": False,
        "use_gaussian": False,
        "use_temporal_short": False,
        "use_temporal_long": True,
        "use_answer_negatives": False,
        "default_alpha": 1.0,
    },
    # Answer/text grounding experiments (V, Q, A) vs (V, Q, ~A)
    "T-01": {
        "description": "Batch answer negatives (baseline)",
        "use_batch_negatives": True,
        "use_blackened": False,
        "use_gaussian": False,
        "use_temporal_short": False,
        "use_temporal_long": False,
        "use_answer_negatives": True,
        "default_alpha": 1.0,
    },
    "T-02": {
        "description": "Temporal answer mismatch (short, ±30s)",
        "use_batch_negatives": True,
        "use_blackened": False,
        "use_gaussian": False,
        "use_temporal_short": True,
        "use_temporal_long": False,
        "use_answer_negatives": True,
        "default_alpha": 1.0,
    },
    "T-03": {
        "description": "Temporal answer mismatch (long, >2min)",
        "use_batch_negatives": True,
        "use_blackened": False,
        "use_gaussian": False,
        "use_temporal_short": False,
        "use_temporal_long": True,
        "use_answer_negatives": True,
        "default_alpha": 1.0,
    },
    # Combined full formula
    "FULL": {
        "description": "Full formula: batch + blackened + temporal negatives",
        "use_batch_negatives": True,
        "use_blackened": True,
        "use_gaussian": False,
        "use_temporal_short": True,
        "use_temporal_long": False,
        "use_answer_negatives": True,
        "default_alpha": 5.0,
    },
    # Custom: user can override via CLI flags
    "CUSTOM": {
        "description": "Custom formula — all flags set via CLI",
        "use_batch_negatives": True,
        "use_blackened": False,
        "use_gaussian": False,
        "use_temporal_short": False,
        "use_temporal_long": False,
        "use_answer_negatives": False,
        "default_alpha": 1.0,
    },
    # V-06: Entity-masked blackened frames — CL loss only on detail tokens
    "V-06": {
        "description": "Blackened frames + entity-masked scoring (detail tokens only)",
        "use_batch_negatives": True,
        "use_blackened": True,
        "use_gaussian": False,
        "use_temporal_short": False,
        "use_temporal_long": False,
        "use_answer_negatives": False,
        "use_entity_masking": True,
        "default_alpha": 5.0,
    },
    # V-07: Anchor-weighted scoring — soft-weight gold MCQ tokens 5×
    # Uses the actual ground-truth MCQ choice text (e.g. "278 yen") rather
    # than regex entity detection. Tokens that overlap the gold-anchor span
    # in the rephrased SFT label receive weight `anchor_weight`; all other
    # answer tokens keep weight 1.0. For FineVideo (no anchor) it falls back
    # to uniform weights == identical to V-02.
    "V-07": {
        "description": "Blackened frames + anchor-weighted scoring (GT MCQ text 5×)",
        "use_batch_negatives": True,
        "use_blackened": True,
        "use_gaussian": False,
        "use_temporal_short": False,
        "use_temporal_long": False,
        "use_answer_negatives": False,
        "use_anchor_weighting": True,
        "anchor_weight": 5.0,
        "default_alpha": 5.0,
    },
    # T-04: V-07 + MCQ-distractor answer negatives.
    # The hardest realistic negative: same video, same prefix, but a wrong
    # MCQ choice substituted into the gold-anchor span.  Distractor scores
    # are computed from the SAME positive forward pass (no extra compute):
    # at each gold-anchor-token position we look up the distractor's token
    # at the same relative index and score log p(distractor|context).
    "T-04": {
        "description": "Blackened + anchor-weighted + MCQ-distractor answer negatives",
        "use_batch_negatives": True,
        "use_blackened": True,
        "use_gaussian": False,
        "use_temporal_short": False,
        "use_temporal_long": False,
        "use_answer_negatives": True,
        "use_anchor_weighting": True,
        "anchor_weight": 5.0,
        "default_alpha": 5.0,
    },
    # T-05: ONLY MCQ-distractor answer negatives — no blackened, no α penalty.
    # Removes the "use the video" pressure that hurts Hallucination/counting
    # while keeping the distractor signal that helps reasoning categories.
    "T-05": {
        "description": "MCQ-distractor answer negatives only (no blackened frames)",
        "use_batch_negatives": False,
        "use_blackened": False,
        "use_gaussian": False,
        "use_temporal_short": False,
        "use_temporal_long": False,
        "use_answer_negatives": True,
        "use_anchor_weighting": True,
        "anchor_weight": 5.0,
        "default_alpha": 1.0,
    },
}


# ═══════════════════════════════════════════════════════════════════════
# Stage 8: Entity-masked scoring (V-06)
# ═══════════════════════════════════════════════════════════════════════

# Patterns for visually-grounded entities — numbers, timestamps, colors
_ENTITY_NUMBER_RE = re.compile(r'\b\d[\d,\.]*\b')
_ENTITY_TIME_RE = re.compile(r'\b\d{1,2}:\d{2}(?::\d{2})?\b')
_ENTITY_COLOR_RE = re.compile(
    r'\b(?:red|blue|green|yellow|orange|purple|pink|black|white|brown|'
    r'gray|grey|cyan|magenta|golden|silver|beige|maroon|navy|teal|violet|'
    r'indigo|rose-red|off-white)\b',
    re.IGNORECASE,
)


def _find_entity_char_spans(text: str) -> List[Tuple[int, int]]:
    """
    Find character-level spans of visually-grounded entities in text.
    Returns list of (start, end) character indices.
    """
    spans = []
    for pattern in [_ENTITY_TIME_RE, _ENTITY_NUMBER_RE, _ENTITY_COLOR_RE]:
        for m in pattern.finditer(text):
            spans.append((m.start(), m.end()))
    # Merge overlapping spans
    if not spans:
        return []
    spans.sort()
    merged = [spans[0]]
    for s, e in spans[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def build_entity_token_mask(
    labels: torch.Tensor,
    tokenizer,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Build a boolean mask over label positions where True = entity token.

    Strategy: decode the answer tokens to text, find entity character spans,
    then map back to token positions by decoding each token and tracking
    cumulative character offsets.

    This runs once per sample per step (not per-negative), so the cost
    of individual token decoding is acceptable.

    Args:
        labels:       [seq_len] label tensor with IGNORE_INDEX for non-answer
        tokenizer:    HF tokenizer for decoding
        ignore_index: value used for masked positions
    Returns:
        [seq_len] boolean tensor, True for entity tokens
    """
    device = labels.device
    mask = torch.zeros(labels.shape[0], dtype=torch.bool, device=device)

    # Extract answer token positions and IDs
    answer_positions = (labels != ignore_index).nonzero(as_tuple=True)[0]
    if answer_positions.numel() == 0:
        return mask

    answer_ids = labels[answer_positions].tolist()

    # Decode full answer text
    full_text = tokenizer.decode(answer_ids, skip_special_tokens=True)
    if not full_text.strip():
        return mask

    # Find entity character spans
    entity_spans = _find_entity_char_spans(full_text)
    if not entity_spans:
        return mask

    # Map tokens to character offsets by decoding each token individually
    # and tracking cumulative position in the full decoded text
    char_offset = 0
    token_char_ranges = []  # (char_start, char_end) per answer token
    for tid in answer_ids:
        token_text = tokenizer.decode([tid], skip_special_tokens=True)
        # Find this token's text in the full text starting from char_offset
        # Use find() to handle tokenizer quirks (spaces, special chars)
        idx = full_text.find(token_text, char_offset)
        if idx >= 0:
            token_char_ranges.append((idx, idx + len(token_text)))
            char_offset = idx + len(token_text)
        else:
            # Token doesn't map cleanly (e.g., subword boundary)
            # Assign it the current offset with zero width
            token_char_ranges.append((char_offset, char_offset))

    # Mark tokens that overlap with any entity span
    for i, (t_start, t_end) in enumerate(token_char_ranges):
        if t_end <= t_start:
            continue
        for e_start, e_end in entity_spans:
            if t_start < e_end and t_end > e_start:  # overlap
                mask[answer_positions[i]] = True
                break

    return mask


def compute_generation_log_likelihood_masked(
    logits: torch.Tensor,
    labels: torch.Tensor,
    entity_mask: torch.Tensor,
    ignore_index: int = -100,
    fallback_to_full: bool = True,
) -> torch.Tensor:
    """
    Compute mean log-probability restricted to entity (detail) tokens only.

    This is the key V-06 innovation: by scoring only tokens that correspond
    to numbers, colors, and named entities, the contrastive gradient flows
    exclusively to the visually-grounded positions, preventing the model
    from satisfying the CL objective through template tokens alone.

    Args:
        logits:       [seq_len, vocab_size] model output logits
        labels:       [seq_len] token IDs with IGNORE_INDEX for non-answer
        entity_mask:  [seq_len] boolean mask, True = entity token
        ignore_index: value used for masked positions in labels
        fallback_to_full: if no entity tokens found, fall back to full scoring
    Returns:
        Scalar: mean log-probability of entity tokens (or all answer tokens
        if no entity tokens and fallback_to_full=True).
    """
    # Shift for next-token prediction
    shift_logits = logits[:-1]       # [seq_len-1, vocab]
    shift_labels = labels[1:]        # [seq_len-1]
    shift_entity = entity_mask[1:]   # [seq_len-1] — shifted to align with labels

    # Combined mask: must be an answer token AND an entity token
    answer_mask = shift_labels != ignore_index
    combined_mask = answer_mask & shift_entity

    if not combined_mask.any():
        if fallback_to_full:
            # No entity tokens in this sample — fall back to full answer scoring
            # (e.g., FineVideo samples without numbers/colors)
            return compute_generation_log_likelihood(logits, labels, ignore_index)
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

    # Compute per-token log probs for entity tokens only
    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(
        dim=-1, index=shift_labels[combined_mask].unsqueeze(-1)
    ).squeeze(-1)

    return token_log_probs.mean()


# ═══════════════════════════════════════════════════════════════════════
# Stage 9: Anchor-weighted scoring (V-07)
# ═══════════════════════════════════════════════════════════════════════

def _normalize_for_match(text: str) -> str:
    """Lowercase + collapse whitespace + strip punctuation for fuzzy substring match."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _find_anchor_char_spans(answer_text: str, anchor_text: str) -> List[Tuple[int, int]]:
    """
    Find character spans in `answer_text` that correspond to the gold MCQ
    anchor.

    Strategy (in order of preference):
      1. Exact (case-insensitive) substring match on the raw text.
      2. Lowercase substring match.
      3. Word-level overlap: for each "content word" (>=3 chars) in the
         normalized anchor, find every occurrence in the answer text.

    Returns a list of (start, end) char ranges in the original answer_text.
    """
    if not anchor_text or not answer_text:
        return []

    # 1. Try direct case-insensitive substring of the FULL anchor
    lower_ans = answer_text.lower()
    lower_anc = anchor_text.lower().strip()

    spans = []
    start = 0
    while True:
        idx = lower_ans.find(lower_anc, start)
        if idx < 0:
            break
        spans.append((idx, idx + len(lower_anc)))
        start = idx + max(1, len(lower_anc))
    if spans:
        return spans

    # 2. Word-level overlap: each anchor content word matched in answer
    norm_anc = _normalize_for_match(anchor_text)
    anchor_words = [w for w in norm_anc.split() if len(w) >= 3]
    # Special-case: short numbers / single letters / colors
    short_anc = [w for w in norm_anc.split() if len(w) < 3 and (w.isdigit() or w.isalpha())]
    anchor_words = anchor_words + short_anc

    if not anchor_words:
        return []

    spans = []
    for w in anchor_words:
        # Match whole-word with simple boundary
        for m in re.finditer(r"\b" + re.escape(w) + r"\b", answer_text, re.IGNORECASE):
            spans.append((m.start(), m.end()))

    if not spans:
        return []

    # Merge overlapping spans
    spans.sort()
    merged = [spans[0]]
    for s, e in spans[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def build_anchor_token_weights(
    labels: torch.Tensor,
    anchor_text: str,
    tokenizer,
    anchor_weight: float = 5.0,
    base_weight: float = 1.0,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Build a per-token float weight tensor over `labels`:
      - Non-answer positions (labels == ignore_index): 0.0  (ignored)
      - Answer tokens overlapping the anchor span:     `anchor_weight`
      - Other answer tokens:                            `base_weight`

    Used by V-07 to amplify the contrastive gradient on tokens that carry
    the actual ground-truth MCQ content (e.g. "278 yen", "Hazardous
    electrical parts may be present") relative to template tokens.

    If no anchor is found (empty anchor_text or no match in the rephrased
    answer), all answer tokens receive `base_weight` — equivalent to
    plain log-likelihood scoring.

    Args:
        labels:        [seq_len] label tensor with ignore_index masked
        anchor_text:   the gold MCQ choice text (e.g. cgbench `answer` field)
        tokenizer:     HF tokenizer for decoding
        anchor_weight: weight for anchor-overlapping tokens
        base_weight:   weight for other answer tokens
        ignore_index:  value used for masked positions
    Returns:
        [seq_len] float tensor of per-position weights
    """
    device = labels.device
    weights = torch.zeros(labels.shape[0], dtype=torch.float32, device=device)

    # Mark all answer tokens with base_weight first
    answer_positions = (labels != ignore_index).nonzero(as_tuple=True)[0]
    if answer_positions.numel() == 0:
        return weights
    weights[answer_positions] = base_weight

    if not anchor_text or not anchor_text.strip():
        return weights

    # Decode the answer text and locate anchor character spans
    answer_ids = labels[answer_positions].tolist()
    full_text = tokenizer.decode(answer_ids, skip_special_tokens=True)
    if not full_text.strip():
        return weights

    anchor_spans = _find_anchor_char_spans(full_text, anchor_text)
    if not anchor_spans:
        return weights

    # Map each answer token to its character range in `full_text`
    char_offset = 0
    token_char_ranges = []
    for tid in answer_ids:
        token_text = tokenizer.decode([tid], skip_special_tokens=True)
        idx = full_text.find(token_text, char_offset) if token_text else -1
        if idx >= 0:
            token_char_ranges.append((idx, idx + len(token_text)))
            char_offset = idx + len(token_text)
        else:
            token_char_ranges.append((char_offset, char_offset))

    # Promote tokens that overlap any anchor span to anchor_weight
    for i, (t_start, t_end) in enumerate(token_char_ranges):
        if t_end <= t_start:
            continue
        for a_start, a_end in anchor_spans:
            if t_start < a_end and t_end > a_start:
                weights[answer_positions[i]] = anchor_weight
                break

    return weights


def compute_generation_log_likelihood_weighted(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Weighted mean of per-token log-probabilities of answer tokens.

    score = (Σᵢ wᵢ · log p(tᵢ)) / (Σᵢ wᵢ)        for i ∈ answer positions

    Args:
        logits:       [seq_len, vocab_size]
        labels:       [seq_len] with ignore_index for non-answer positions
        weights:      [seq_len] float weights, 0.0 for non-answer positions
        ignore_index: value for masked positions in labels
    Returns:
        scalar weighted mean log-prob of answer tokens
    """
    shift_logits  = logits[:-1]
    shift_labels  = labels[1:]
    shift_weights = weights[1:]

    answer_mask = shift_labels != ignore_index
    if not answer_mask.any():
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

    sel_logits  = shift_logits[answer_mask]
    sel_labels  = shift_labels[answer_mask]
    sel_weights = shift_weights[answer_mask].to(sel_logits.dtype)

    log_probs = torch.nn.functional.log_softmax(sel_logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=sel_labels.unsqueeze(-1)).squeeze(-1)

    weight_sum = sel_weights.sum()
    if weight_sum <= 0:
        # No active weights — fall back to uniform mean
        return token_log_probs.mean()

    return (token_log_probs * sel_weights).sum() / weight_sum


# ═══════════════════════════════════════════════════════════════════════
# Stage 9b: MCQ-distractor answer negatives (T-04)
# ═══════════════════════════════════════════════════════════════════════

def compute_distractor_negative_score(
    logits: torch.Tensor,
    labels: torch.Tensor,
    gold_anchor_text: str,
    distractor_anchor_text: str,
    tokenizer,
    anchor_weight: float = 5.0,
    base_weight: float = 1.0,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Score the model's confidence in a WRONG MCQ choice using the SAME
    forward pass as the positive.

    Construction:
      - Take the gold answer-token sequence (from `labels`)
      - Locate the gold-anchor character span (e.g. "corn" within
        "...vegetables on the book are corn...")
      - Encode the distractor text (e.g. "beans") into tokens
      - For each gold-anchor token at position k, REPLACE the target
        with `distractor_token_ids[k]` (truncated when distractor is
        shorter; trailing distractor tokens dropped when longer)
      - Compute weighted log-likelihood under the same logits

    Returns a scalar score on the SAME scale as
    `compute_generation_log_likelihood_weighted` so it can plug
    straight into the InfoNCE denominator.

    If no anchor is found (e.g. FineVideo, malformed sample), returns
    a "neutral" score equal to the unweighted positive log-likelihood —
    contributing zero gradient pressure for that sample.
    """
    device = logits.device
    answer_positions = (labels != ignore_index).nonzero(as_tuple=True)[0]
    if answer_positions.numel() == 0:
        return torch.tensor(0.0, device=device, dtype=logits.dtype)

    answer_ids = labels[answer_positions].tolist()
    full_text = tokenizer.decode(answer_ids, skip_special_tokens=True)

    # ── Find gold-anchor character span(s) ──
    if not gold_anchor_text or not gold_anchor_text.strip() \
            or not distractor_anchor_text or not distractor_anchor_text.strip():
        # Degenerate case: no anchor or no distractor → return plain LL
        return compute_generation_log_likelihood(
            logits, labels, ignore_index=ignore_index,
        )

    anchor_spans = _find_anchor_char_spans(full_text, gold_anchor_text)
    if not anchor_spans:
        return compute_generation_log_likelihood(
            logits, labels, ignore_index=ignore_index,
        )

    # ── Map each gold-answer token → its char range in `full_text` ──
    char_offset = 0
    token_char_ranges = []
    for tid in answer_ids:
        token_text = tokenizer.decode([tid], skip_special_tokens=True)
        idx = full_text.find(token_text, char_offset) if token_text else -1
        if idx >= 0:
            token_char_ranges.append((idx, idx + len(token_text)))
            char_offset = idx + len(token_text)
        else:
            token_char_ranges.append((char_offset, char_offset))

    # ── Identify gold-anchor token indices (within `answer_positions`) ──
    anchor_token_indices = []
    for i, (t_start, t_end) in enumerate(token_char_ranges):
        if t_end <= t_start:
            continue
        for a_start, a_end in anchor_spans:
            if t_start < a_end and t_end > a_start:
                anchor_token_indices.append(i)
                break

    if not anchor_token_indices:
        return compute_generation_log_likelihood(
            logits, labels, ignore_index=ignore_index,
        )

    # ── Tokenize distractor anchor (with leading space for proper BPE) ──
    distractor_ids = tokenizer.encode(
        " " + distractor_anchor_text.strip(),
        add_special_tokens=False,
    )
    if not distractor_ids:
        return compute_generation_log_likelihood(
            logits, labels, ignore_index=ignore_index,
        )

    # ── Build modified labels: gold everywhere EXCEPT anchor positions
    #    where gold-anchor-token k is replaced by distractor_ids[k]
    #    (truncate when distractor shorter; drop extra distractor tokens
    #    when longer — we only score positions we have logits for) ──
    modified_labels = labels.clone()
    n_replace = min(len(anchor_token_indices), len(distractor_ids))
    for k in range(n_replace):
        token_idx_in_answer = anchor_token_indices[k]
        seq_pos = answer_positions[token_idx_in_answer].item()
        modified_labels[seq_pos] = int(distractor_ids[k])

    # ── Build weights: anchor_weight on the (now-distractor) anchor
    #    positions, base_weight elsewhere — matches positive's weighting ──
    weights = torch.zeros(labels.shape[0], dtype=torch.float32, device=device)
    weights[answer_positions] = base_weight
    for k in range(n_replace):
        token_idx_in_answer = anchor_token_indices[k]
        seq_pos = answer_positions[token_idx_in_answer].item()
        weights[seq_pos] = anchor_weight

    return compute_generation_log_likelihood_weighted(
        logits, modified_labels, weights, ignore_index=ignore_index,
    )


def get_experiment_config(experiment_id: str) -> dict:
    """
    Look up the experiment configuration by ID.
    
    Args:
        experiment_id: One of V-01..V-05, T-01..T-03, FULL, CUSTOM
    Returns:
        Dict with boolean flags for which negative strategies to use.
    """
    if experiment_id not in EXPERIMENT_CONFIGS:
        raise ValueError(
            f"Unknown experiment_id '{experiment_id}'. "
            f"Valid options: {list(EXPERIMENT_CONFIGS.keys())}"
        )
    return EXPERIMENT_CONFIGS[experiment_id].copy()
