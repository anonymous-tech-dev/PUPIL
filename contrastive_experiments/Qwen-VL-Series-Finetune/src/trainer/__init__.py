try:
    from src.trainer.contrastive_sft_trainer import ContrastiveSFTTrainer
except ImportError:
    ContrastiveSFTTrainer = None

# Optional imports — these trainers may not be present in all clones
try:
    from src.trainer.dpo_trainer import QwenDPOTrainer
except ImportError:
    QwenDPOTrainer = None

try:
    from src.trainer.sft_trainer import QwenSFTTrainer, GenerativeEvalPrediction
except ImportError:
    QwenSFTTrainer = None
    GenerativeEvalPrediction = None

try:
    from src.trainer.grpo_trainer import QwenGRPOTrainer
except ImportError:
    QwenGRPOTrainer = None

try:
    from src.trainer.cls_trainer import QwenCLSTrainer
except ImportError:
    QwenCLSTrainer = None

__all__ = [
    "ContrastiveSFTTrainer",
    "QwenSFTTrainer",
    "QwenDPOTrainer",
    "QwenGRPOTrainer",
    "QwenCLSTrainer",
    "GenerativeEvalPrediction",
]
