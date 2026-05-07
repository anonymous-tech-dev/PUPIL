"""GPT-5.4 / 5.1 via Azure Azure in *blind* mode (no video frames, no transcript).

Mirrors `gpt54_transcript.py` end-to-end (same deployments, same azure_ai/GCR
fallback, same retry policy, same reasoning-effort knob), but the user message
contains ONLY the question — no images and no transcript.  Used by the
"how much can the model solve from prior knowledge alone?" ablation.
"""

import os
import time

from azure.identity import AzureCliCredential, get_bearer_token_provider
from openai import AzureOpenAI

from models.base import BaseEvaluator
from models.blind_base import BLIND_SYSTEM, build_blind_prompt


class GPT54BlindEvaluator(BaseEvaluator):
    """GPT-5.4 blind / no-context baseline."""

    DEPLOYMENT_NAME = "gpt-5.4_2026-03-05"

    def load(self):
        self.deployment_name        = self.DEPLOYMENT_NAME
        # Azure accepts {none, minimal, low, medium, high, xhigh}.
        # The blind run is a *prior-knowledge-only* baseline; we default to
        # 'none' so the score reflects raw recall rather than test-time
        # reasoning over the (absent) context. Override with
        # GPT_REASONING_EFFORT=high if you want a thinking-blind variant.
        self.reasoning_effort       = os.environ.get(
            "GPT_REASONING_EFFORT", "none"
        ).strip().lower()
        if self.reasoning_effort in {"", "auto", "default"}:
            self.reasoning_effort = "none"
        # Blind answers are pure language — 1024 tokens is plenty by default,
        # but allow override (mirroring the transcript / video evaluators).
        self.max_completion_tokens  = int(os.environ.get(
            "GPT_MAX_COMPLETION_TOKENS", "1024"
        ))
        self.max_retries            = 3
        self.retry_delay            = 5

        credential = AzureCliCredential()
        token_provider = get_bearer_token_provider(
            credential, "api://azure/.default"
        )
        self.client_azure_ai = AzureOpenAI(
            azure_endpoint="https://<AZURE_OPENAI_ENDPOINT>",
            azure_ad_token_provider=token_provider,
            api_version="2024-12-01-preview",
        )
        self.client_azure_ai = AzureOpenAI(
            azure_endpoint="https://<AZURE_OPENAI_ENDPOINT>",
            azure_ad_token_provider=token_provider,
            api_version="2024-12-01-preview",
        )
        self.active_client = self.client_azure_ai
        self._consecutive_rate_limits = 0

    def _call_with_fallback(self, messages):
        kwargs = dict(
            model=self.deployment_name,
            messages=messages,
            max_completion_tokens=self.max_completion_tokens,
        )
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort

        for attempt in range(self.max_retries):
            try:
                response = self.active_client.chat.completions.create(**kwargs)
                self._consecutive_rate_limits = 0
                return response.choices[0].message.content
            except Exception as e:
                err = str(e).lower()
                if "rate" in err or "429" in err or "throttl" in err:
                    self._consecutive_rate_limits += 1
                    if self._consecutive_rate_limits >= 3:
                        self.active_client = (
                            self.client_azure_ai
                            if self.active_client is self.client_azure_ai
                            else self.client_azure_ai
                        )
                        self._consecutive_rate_limits = 0
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                    else:
                        raise
        return "ERROR: All retries exhausted"

    def generate_response(self, video_path, prompt):
        # video_path is intentionally unused — this is the *blind* baseline.
        del video_path
        user_text = build_blind_prompt(prompt)
        messages = [
            {"role": "system", "content": BLIND_SYSTEM},
            {"role": "user",   "content": user_text},
        ]
        return self._call_with_fallback(messages)


class GPT51BlindEvaluator(GPT54BlindEvaluator):
    """GPT-5.1 blind variant (same plumbing, older deployment)."""

    DEPLOYMENT_NAME = "gpt-5.1_2025-11-13"

    def load(self):
        super().load()
        self.deployment_name = self.DEPLOYMENT_NAME
