import gc
import torch

class BaseEvaluator:
    def __init__(self, device="cuda", dtype=torch.bfloat16):
        self.device = device
        self.dtype = dtype
        self.model = None
        self.processor = None

    def load(self):
        raise NotImplementedError

    def generate_response(self, video_path: str, prompt: str) -> str:
        raise NotImplementedError

    def unload(self):
        if self.model is not None:
            del self.model
        if self.processor is not None:
            del self.processor
        self.model = None
        self.processor = None
        gc.collect()
        torch.cuda.empty_cache()