from dataclasses import dataclass, field
from typing import Optional

try:
    from accelerate.utils import ParallelismConfig as _PC
except Exception:
    class _PC:
        pass

import transformers.training_args as _ta
if not hasattr(_ta, "ParallelismConfig"):
    _ta.ParallelismConfig = _PC

from transformers import TrainingArguments as HFTrainingArguments
from trl import DPOConfig as DPOConfigTRL
from trl import GRPOConfig as GRPOConfigTRL


@dataclass
class ModelArguments:
    model_id: Optional[str] = field(default="Qwen/Qwen2-VL-7B-Instruct")
    # SoF-DPO warm-start: path to a PEFT/LoRA adapter dir produced by SFT.
    # If set, the adapter is loaded onto the base model and `merge_and_unload`d
    # into the weights BEFORE the new (DPO) LoRA is attached. The same merge
    # is applied to ref_model when one is constructed (lora_enable=False), so
    # the DPO KL reference is the SFT policy, not the raw base.
    sft_adapter_path: Optional[str] = field(default=None)


@dataclass
class CLSArguments(HFTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    adam_epsilon: float = field(default=1e-8)

    freeze_vision_tower: bool = field(default=False)
    freeze_llm: bool = field(default=False)
    freeze_merger: bool = field(default=False)
    disable_flash_attn2: bool = field(default=False)
    unfreeze_topk_llm: int = 0
    unfreeze_topk_vision: int = 0
    mlp_head_dim: Optional[int] = field(default=0)
    mlp_head_dropout: Optional[float] = field(default=0.0)
    
    loss_type : str = field(
        default="cross_entropy",
        metadata={"help": "Loss type to use. Should be one of `cross_entropy`, `focal_loss`, `class_balanced_cross_entropy`, or `class_balanced_focal_loss`."}
    )
    focal_alpha: Optional[str] = field(
        default=None,
        metadata={"help": "Focal Loss alpha value. If None use CrossEntropyLoss. ex '1.0,7.5'"}
    )
    focal_gamma: float = field(
        default=0.0,
        metadata={"help": "Focal Loss gamma value"}
    )
    num_labels: int = field(
        default=2,
        metadata={"help": "Number of labels for classification."}
    )
    class_balanced_beta: float = field(
        default=0.999,
        metadata={"help": "Beta value for Class Balanced Loss. If 0.0, use standard CrossEntropyLoss."}
    )
    early_stopping_patience: int = field(
        default=0,
        metadata={"help": "Number of epochs with no improvement after which training will be stopped."}
    )
    early_stopping_threshold: float = field(
        default=0.0,
        metadata={"help": "Minimum change in the monitored quantity to qualify as an improvement."}
    )

    max_seq_length: int = field(
        default=32768, # This is the default value of the qwen2-vl model
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    vision_lora: bool = False
    use_dora: bool = False
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    vision_lr: Optional[float] = None
    merger_lr: Optional[float] = None
    head_lr: Optional[float] = None
    lora_namespan_exclude: str = field(default=None, metadata={"help": "List of namespan to exclude for LoRA"})
    num_lora_modules: int = -1
    use_liger_kernel: bool = True


@dataclass
class TrainingArguments(HFTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    adam_epsilon: float = field(default=1e-8)

    freeze_vision_tower: bool = field(default=False)
    freeze_llm: bool = field(default=False)
    freeze_merger: bool = field(default=False)
    disable_flash_attn2: bool = field(default=False)
    unfreeze_topk_llm: int = 0
    unfreeze_topk_vision: int = 0

    max_seq_length: int = field(
        default=32768, # This is the default value of the qwen2-vl model
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )

    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    vision_lora: bool = False
    use_dora: bool = False
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    vision_lr: Optional[float] = None
    merger_lr: Optional[float] = None
    lora_namespan_exclude: str = field(default=None, metadata={"help": "List of namespan to exclude for LoRA"})
    num_lora_modules: int = -1
    use_liger_kernel: bool = True

    # Generation-based evaluation settings
    generation_max_new_tokens: int = field(
        default=512,
        metadata={"help": "Maximum number of new tokens to generate during evaluation."}
    )

@dataclass
class DPOArguments(DPOConfigTRL):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    adam_epsilon: float = field(default=1e-8)

    freeze_vision_tower: bool = field(default=False)
    freeze_llm: bool = field(default=False)
    freeze_merger: bool = field(default=False)
    disable_flash_attn2: bool = field(default=False)
    unfreeze_topk_llm: int = 0
    unfreeze_topk_vision: int = 0

    max_seq_length: int = field(
        default=32768, # This is the default value of the qwen2-vl model
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    vision_lora: bool = False
    use_dora: bool = False
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    vision_lr: Optional[float] = None
    merger_lr: Optional[float] = None
    lora_namespan_exclude: str = field(default=None, metadata={"help": "List of namespan to exclude for LoRA"})
    num_lora_modules: int = -1
    use_liger_loss: bool = True
    beta: float = field(
        default=0.1,
        metadata={"help": "The beta value for DPO."}
    )
    precompute_ref_log_probs: bool = field(
        default=False,
        metadata={"help": "Whether to precompute the reference log probabilities."}
    )
    dpo_loss:str = field(
        default="sigmoid",
        metadata={"help": "The type of DPO loss to use."}
    )

@dataclass
class GRPOArguments(GRPOConfigTRL):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    adam_epsilon: float = field(default=1e-8)

    freeze_vision_tower: bool = field(default=False)
    freeze_llm: bool = field(default=False)
    freeze_merger: bool = field(default=False)
    disable_flash_attn2: bool = field(default=False)
    unfreeze_topk_llm: int = 0
    unfreeze_topk_vision: int = 0

    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    vision_lora: bool = False
    use_dora: bool = False
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    vision_lr: Optional[float] = None
    merger_lr: Optional[float] = None
    lora_namespan_exclude: str = field(default=None, metadata={"help": "List of namespan to exclude for LoRA"})
    num_lora_modules: int = -1
    beta: float = field(
        default=0.04,
        metadata={
            "help": "KL coefficient. If `0.0`, the reference model is not loaded, reducing memory usage and improving "
            "training speed, but may be numerically unstable for long training runs."
        },
    )
    temperature: float = 0.9
    top_p: float = 1.0
    top_k: int = 50
    min_p: Optional[float] = None
    repetition_penalty: float = 1.0
    max_completion_length: int = 256
    max_prompt_length: int = 512
    use_liger_loss: bool = True


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_path: str= field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    eval_image_folder: Optional[str] = field(
        default=None, metadata={"help": "Path to the evaluation image data."}
    )
    lazy_preprocess: bool = False
    image_folder: Optional[str] = field(default=None)
    image_min_pixels: Optional[int] = field(default=3136)
    image_max_pixels: Optional[int] = field(default=12845056)
    video_min_pixels: Optional[int] = field(default=100352)
    video_max_pixels: Optional[int] = field(default=602112)
    image_resized_width: int = field(default=None)
    image_resized_height: int = field(default=None)
    video_resized_width: int = field(default=None)
    video_resized_height: int = field(default=None)
    fps: Optional[int] = field(default=None, metadata={"help": "Frames per second for video data."})
    nframes: Optional[int] = field(default=None, metadata={"help": "Number of frames for video data."})
    video_max_frames: Optional[int] = field(
        default=None,
        metadata={"help": "Hard cap on sampled frames per video. "
                  "Auto-computed from max_seq_length if not set."}
    )
    video_total_pixels: Optional[int] = field(
        default=None,
        metadata={"help": "Total pixel budget for the whole video (controls "
                  "dynamic per-frame resolution scaling in qwen_vl_utils). "
                  "Auto-computed from max_seq_length if not set."}
    )


"""
Contrastive Learning Arguments for Qwen3VL Fine-tuning
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ContrastiveArguments:
    """
    Arguments for contrastive learning regularization.
    
    These parameters control the InfoNCE-based multimodal contrastive loss
    that helps ground the model's answers in visual context and prevents
    text-only hallucinations.
    """
    
    # Core contrastive parameters
    contrastive_weight: float = field(
        default=0.1,
        metadata={"help": "Weight (λ) for contrastive loss in total loss: L_total = L_sft + λ * L_contrastive"}
    )
    
    contrastive_temperature: float = field(
        default=0.07,
        metadata={"help": "Temperature (τ) for InfoNCE loss. Lower values make the model more confident."}
    )
    
    # Negative sampling strategy
    negative_strategy: str = field(
        default="batch",
        metadata={
            "help": (
                "Strategy for sampling negatives. Options:\n"
                "  - 'batch': Standard InfoNCE with batch negatives (baseline)\n"
                "  - 'blackened': Use blackened frames for grounding penalty (V-02)\n"
                "  - 'gaussian': Use Gaussian noise frames (V-03)\n"
                "  - 'temporal': Use temporal negatives (same video, different time)\n"
                "  - 'hard_negatives': Mine hard negatives from same video\n"
                "  - 'all': Combine all strategies"
            )
        }
    )
    
    num_negatives: int = field(
        default=4,
        metadata={"help": "Number of negative samples (K) per positive sample"}
    )
    
    # Grounding penalty
    alpha_grounding_penalty: float = field(
        default=5.0,
        metadata={
            "help": (
                "Penalty weight (α) for blackened frames. Higher values penalize "
                "the model more heavily for matching text to corrupted visual input."
            )
        }
    )
    
    # Hard negative mining
    use_hard_negatives: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to mine hard negatives from the same video. "
                "Exploits the 10-QA-per-video structure to create pairs like "
                "(same video, wrong question) and (same video, wrong answer)."
            )
        }
    )
    
    hard_negative_ratio: float = field(
        default=0.5,
        metadata={
            "help": (
                "Ratio of hard negatives to random negatives. "
                "0.5 means 50% hard negatives, 50% random negatives."
            )
        }
    )
    
    # Memory queue (MoCo-style)
    use_memory_queue: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use a memory queue for storing negative samples. "
                "Increases the effective batch size for contrastive learning "
                "without requiring more GPU memory."
            )
        }
    )
    
    queue_size: int = field(
        default=65536,
        metadata={"help": "Size of the memory queue (if use_memory_queue=True)"}
    )
    
    # Multi-positive contrastive learning
    use_multi_positive: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use multi-positive contrastive learning. "
                "Treats multiple QA pairs from the same video as positives. "
                "Useful for learning that different questions about the same "
                "video should have embeddings that cluster together."
            )
        }
    )
    
    # Temporal contrastive learning
    temporal_window_size: int = field(
        default=300,
        metadata={
            "help": (
                "Temporal window size (in seconds) for temporal negatives. "
                "Frames outside this window (±temporal_window_size) from the "
                "ground truth frame are considered temporal negatives."
            )
        }
    )
    
    # Hierarchical contrastive learning
    use_hierarchical_loss: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use hierarchical contrastive loss. "
                "Applies contrastive learning at multiple levels: "
                "video-level, scene-level, and frame-level."
            )
        }
    )
    
    # ── Contrastive mode knob ──
    contrastive_mode: str = field(
        default="generative",
        metadata={
            "help": (
                "Similarity scoring method for InfoNCE loss.\n"
                "  'generative': compute standard next-token cross-entropy loss "
                "on [Video+Question+Answer], contrast log-likelihood of true "
                "video vs corrupted/other videos.\n"
                "  'vector': run separate forward passes with output_hidden_states=True, "
                "extract EOS hidden state, project through linear layer, compute "
                "cosine similarity (Amazon paper style)."
            )
        }
    )

    # ── Data path knobs ──
    cgbench_train_vids_dir: str = field(
        default="/data/Pupil/CGBench/train_vids",
        metadata={
            "help": (
                "Path to CGBench full-length training videos. "
                "Used for temporal negative clip extraction (V-04, V-05)."
            )
        }
    )

    cgbench_anchors_path: str = field(
        default="",
        metadata={
            "help": (
                "Path to cgbench.json containing the original MCQ choices. "
                "Used by V-07 to look up the gold answer text per qid and "
                "build anchor-weighted token weights for CL scoring. "
                "Leave empty to disable anchor weighting."
            )
        }
    )

    # ── Per-source sample count knobs ──
    max_samples_cgbench: int = field(
        default=-1,
        metadata={"help": "Max training samples from CGBench (-1 = use all)."}
    )
    max_samples_finevideo: int = field(
        default=-1,
        metadata={"help": "Max training samples from FineVideo (-1 = use all)."}
    )
    max_samples_edubench: int = field(
        default=-1,
        metadata={"help": "Max training samples from EduBench (-1 = use all)."}
    )
    max_val_samples: int = field(
        default=-1,
        metadata={"help": "Max validation samples per source (-1 = use all)."}
    )

    # ── Reasoning traces knob ──
    use_reasoning_traces: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to append reasoning traces to the better answer "
                "for CGBench data. When True, richer answers with chain-of-thought "
                "are used; when False, only the concise answer is used."
            )
        }
    )

    # ── Number of temporal negative clips per sample ──
    num_temporal_clips: int = field(
        default=1,
        metadata={
            "help": (
                "Number of temporal negative clips to extract per sample. "
                "When >1, multiple non-overlapping shifted clips are sampled "
                "from the same video, giving richer same-video contrastive signal "
                "without wasting batch slots on unrelated videos."
            )
        }
    )

    # ── Gradient flow through negatives ──
    grad_through_negatives: bool = field(
        default=True,
        metadata={
            "help": (
                "When True, negative forward passes run WITH gradients so the "
                "InfoNCE loss can push negative scores DOWN (not just positive UP). "
                "Requires more VRAM but fixes a fundamental issue where detached "
                "negatives make the CL loss equivalent to a reweighted SFT loss."
            )
        }
    )

    # Logging
    log_contrastive_metrics: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to log detailed contrastive learning metrics "
                "(positive similarities, negative similarities, etc.)"
            )
        }
    )
    
    # Debugging
    debug_negatives: bool = field(
        default=False,
        metadata={
            "help": (
                "Enable debugging mode to visualize negative samples. "
                "Saves examples of positive pairs and negative samples to disk."
            )
        }
    )
    
    def __post_init__(self):
        """Validate arguments."""
        # Validate contrastive mode
        valid_modes = ["generative", "vector"]
        if self.contrastive_mode not in valid_modes:
            raise ValueError(
                f"contrastive_mode must be one of {valid_modes}, "
                f"got {self.contrastive_mode}"
            )

        # Validate negative strategy — now uses experiment IDs from contrastive_utils
        valid_strategies = [
            "V-01", "V-02", "V-03", "V-04", "V-05", "V-06", "V-07",
            "T-01", "T-02", "T-03", "T-04", "T-05",
            "FULL", "CUSTOM",
            # Legacy aliases kept for backwards compat
            "batch", "blackened", "gaussian", "temporal", "hard_negatives", "all",
        ]
        if self.negative_strategy not in valid_strategies:
            raise ValueError(
                f"negative_strategy must be one of {valid_strategies}, "
                f"got {self.negative_strategy}"
            )
        
        # Validate temperature
        if self.contrastive_temperature <= 0:
            raise ValueError(f"contrastive_temperature must be positive, got {self.contrastive_temperature}")
        
        # Validate num_negatives
        if self.num_negatives < 1:
            raise ValueError(f"num_negatives must be at least 1, got {self.num_negatives}")
        
        # Validate num_temporal_clips
        if self.num_temporal_clips < 1:
            raise ValueError(f"num_temporal_clips must be at least 1, got {self.num_temporal_clips}")

        # Validate alpha_grounding_penalty
        if self.alpha_grounding_penalty < 0:
            raise ValueError(f"alpha_grounding_penalty must be non-negative, got {self.alpha_grounding_penalty}")
        
        # Validate hard_negative_ratio
        if not 0 <= self.hard_negative_ratio <= 1:
            raise ValueError(f"hard_negative_ratio must be in [0, 1], got {self.hard_negative_ratio}")
        
        # Validate queue_size
        if self.use_memory_queue and self.queue_size < 1:
            raise ValueError(f"queue_size must be at least 1, got {self.queue_size}")


