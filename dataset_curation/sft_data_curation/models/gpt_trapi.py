import os
import cv2
import base64
import numpy as np
from typing import Optional

from azure.identity import AzureCliCredential, get_bearer_token_provider
from openai import AzureOpenAI
from decord import VideoReader, cpu

from models.base import BaseGenerator


class GPTAzureGenerator(BaseGenerator):
    """
    Azure Azure-hosted GPT model (GPT-5, GPT-4o, etc.).
    Auth via AzureCliCredential – run `az login` before use.
    """

    # ------------------------------------------------------------------ #
    #  Knobs (overridable via subclass or monkey-patch before load())      #
    # ------------------------------------------------------------------ #
    DEPLOYMENT_NAME: str = "gpt-5.1_2025-11-13"  # change to any Azure deployment
    FRAMES_TO_EXTRACT: int = 16                   # frames sampled from clue vid
    MAX_DIM: int = 768                            # longest edge after resize
    MAX_COMPLETION_TOKENS: int = 1024

    def load(self):
        print(f"[GPT-Azure] Authenticating → deployment: {self.DEPLOYMENT_NAME}")
        credential = AzureCliCredential()
        token_provider = get_bearer_token_provider(credential, "api://azure/.default")
        self.client = AzureOpenAI(
            azure_endpoint="https://<AZURE_OPENAI_ENDPOINT>",
            azure_ad_token_provider=token_provider,
            api_version="2024-10-21",
        )
        print("[GPT-Azure] Client ready.")

    # ------------------------------------------------------------------ #
    #  Video helpers                                                       #
    # ------------------------------------------------------------------ #
    def _extract_frames(self, video_path: str) -> list[str]:
        """Return list of base64-encoded JPEG frames."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        try:
            vr = VideoReader(video_path, ctx=cpu(0))
        except Exception as e:
            raise RuntimeError(f"decord failed to open {video_path}: {e}") from e

        total_frames = len(vr)
        if total_frames == 0:
            return []

        indices = np.linspace(0, total_frames - 1, self.FRAMES_TO_EXTRACT, dtype=int)
        frames = vr.get_batch(indices).asnumpy()  # (N, H, W, 3) RGB

        encoded: list[str] = []
        for frame in frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            h, w = frame_bgr.shape[:2]
            if max(h, w) > self.MAX_DIM:
                scale = self.MAX_DIM / max(h, w)
                frame_bgr = cv2.resize(
                    frame_bgr,
                    (int(w * scale), int(h * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            encoded.append(base64.b64encode(buf).decode("utf-8"))

        return encoded

    # ------------------------------------------------------------------ #
    #  Core generation                                                     #
    # ------------------------------------------------------------------ #
    def generate_response(
        self,
        prompt: str,
        video_path: Optional[str] = None,
    ) -> str:
        content: list[dict] = [{"type": "text", "text": prompt}]

        if video_path is not None:
            frames = self._extract_frames(video_path)
            for b64 in frames:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}",
                            "detail": "auto",
                        },
                    }
                )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert video-question-answering assistant that produces "
                    "high-quality, well-grounded answers for supervised fine-tuning datasets."
                ),
            },
            {"role": "user", "content": content},
        ]

        response = self.client.chat.completions.create(
            model=self.DEPLOYMENT_NAME,
            messages=messages,
            max_completion_tokens=self.MAX_COMPLETION_TOKENS,
        )
        return response.choices[0].message.content.strip()