"""
stages/clip_selection.py — CLIP-based frame selection (fast, online replacement
for VLM selection).

Instead of 12 expensive autoregressive VLM calls (~200s/question), this:
  1. Encodes all candidate frames with CLIP vision encoder  (batch GPU, ~2s)
  2. Encodes question+choices with CLIP text encoder        (instant)
  3. Ranks frames by cosine similarity
  4. Picks top-m frames

This is the "MAD for frame selection" — replaces the slow VLM-based
selection with a fast heuristic that's still question-aware.

Model: SigLIP-SO400M (via open_clip) — best open CLIP variant for retrieval.
Falls back to ViT-B-32 if SO400M not cached.
"""

import logging
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
from PIL import Image
from omegaconf import DictConfig

logger = logging.getLogger("educot.clip_selection")

FrameBundle = List[Tuple[int, Image.Image]]

# ─── Singleton CLIP model (loaded once, reused) ──────────────────────────
_clip_model = None
_clip_preprocess = None
_clip_tokenizer = None
_clip_device = None


def _load_clip(device: str = "cuda"):
    """Load CLIP model (singleton)."""
    global _clip_model, _clip_preprocess, _clip_tokenizer, _clip_device

    if _clip_model is not None:
        return

    import open_clip

    # Try SigLIP first (best retrieval), fall back to ViT-B-32
    for model_name, pretrained in [
        ("ViT-SO400M-14-SigLIP-384", "webli"),
        ("ViT-B-32", "laion2b_s34b_b79k"),
    ]:
        try:
            logger.info("[CLIP] Loading %s (%s) …", model_name, pretrained)
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained, device=device,
            )
            tokenizer = open_clip.get_tokenizer(model_name)
            model.eval()
            _clip_model = model
            _clip_preprocess = preprocess
            _clip_tokenizer = tokenizer
            _clip_device = device
            logger.info("[CLIP] Ready: %s on %s", model_name, device)
            return
        except Exception as e:
            logger.warning("[CLIP] Failed to load %s: %s", model_name, e)
            continue

    raise RuntimeError("Could not load any CLIP model")


def _encode_frames(images: List[Image.Image], batch_size: int = 64) -> np.ndarray:
    """Encode images → L2-normalised embeddings. Shape: (N, D)."""
    all_embs = []
    for i in range(0, len(images), batch_size):
        batch_imgs = images[i : i + batch_size]
        tensors = torch.stack([_clip_preprocess(img) for img in batch_imgs])
        tensors = tensors.to(_clip_device)
        with torch.no_grad(), torch.amp.autocast(_clip_device):
            embs = _clip_model.encode_image(tensors)
            embs = embs / embs.norm(dim=-1, keepdim=True)
        all_embs.append(embs.cpu().numpy())
    return np.concatenate(all_embs, axis=0)


def _encode_text(text: str) -> np.ndarray:
    """Encode text → L2-normalised embedding. Shape: (1, D)."""
    tokens = _clip_tokenizer([text]).to(_clip_device)
    with torch.no_grad(), torch.amp.autocast(_clip_device):
        emb = _clip_model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()


# ─── Public API ───────────────────────────────────────────────────────────

def clip_selection_call(
    frames: FrameBundle,
    question: str,
    answer_choices: List[str],
    cfg: DictConfig,
    top_k: int = 0,
) -> Dict[str, Any]:
    """
    Rank frames by CLIP similarity to question and select top-k.

    Args:
        frames       : list of (frame_id, PIL.Image)
        question     : the question text
        answer_choices: list of answer choice strings
        cfg          : full DictConfig
        top_k        : how many frames to keep (0 = use context budget)

    Returns:
        {
          "selected_ids"  : List[int],
          "justification" : str,
          "raw_response"  : str,        # similarity scores as string
        }
    """
    _load_clip(device="cuda")

    if not frames:
        return {"selected_ids": [], "justification": "no frames", "raw_response": ""}

    frame_ids = [fid for fid, _ in frames]
    images = [img for _, img in frames]

    # Build query: question + all answer choices
    letters = "ABCDE"
    query_parts = [question]
    for i, c in enumerate(answer_choices):
        query_parts.append(f"({letters[i]}) {c}")
    query = " ".join(query_parts)

    # Encode
    frame_embs = _encode_frames(images)        # (N, D)
    text_emb = _encode_text(query)             # (1, D)

    # Cosine similarity (both already L2-normalised)
    sims = (frame_embs @ text_emb.T).squeeze()  # (N,)

    # Select top-k
    if top_k <= 0:
        top_k = cfg.aggregation.context_budget_frames - cfg.aggregation.uniform_context_frames
    top_k = min(top_k, len(frame_ids))

    top_indices = np.argsort(sims)[::-1][:top_k]
    selected_ids = sorted([frame_ids[i] for i in top_indices])

    # Build justification
    mean_sim = float(sims[top_indices].mean())
    justification = f"CLIP top-{top_k}: mean_sim={mean_sim:.4f}"

    return {
        "selected_ids": selected_ids,
        "justification": justification,
        "raw_response": f"sims_range=[{sims.min():.4f}, {sims.max():.4f}]",
    }


def clip_batch_selection(
    video_path: str,
    candidate_ids: List[int],
    question: str,
    answer_choices: List[str],
    meta,  # VideoMeta
    cfg: DictConfig,
) -> Dict[str, Any]:
    """
    Full CLIP selection on candidate frames.
    
    For efficiency, we subsample candidates to at most clip_pool_size
    frames before decoding + CLIP encoding.  This keeps the total
    decode + encode time under ~10s even for 1-hour videos.
    
    Then we pick top-k by CLIP similarity to the question.
    """
    from stages.video_loading import decode_frames, uniform_subsample_ids

    _load_clip(device="cuda")

    if not candidate_ids:
        return {"selected_ids": [], "justification": "no candidates", "raw_response": ""}

    # Subsample candidates to a manageable pool for CLIP encoding
    clip_pool = getattr(cfg.selection, 'clip_pool_size', 512)
    pool_ids = uniform_subsample_ids(candidate_ids, clip_pool)

    # Decode only the pool frames
    frames = decode_frames(video_path, pool_ids, meta.native_fps, meta.target_fps)
    return clip_selection_call(frames, question, answer_choices, cfg)
