"""GPT-5.4 via Azure Azure on TRANSCRIPT only (no video frames).

Mirrors `gpt54.py` end-to-end (same deployment, same azure_ai/GCR fallback, same
retry policy, same reasoning-effort knob), but the user message contains the
SRT transcript text instead of base64 frames. Used by the
"does our benchmark really need the video?" ablation.
"""

import os
import time

from azure.identity import AzureCliCredential, get_bearer_token_provider
from openai import AzureOpenAI

from models.base import BaseEvaluator
from models.transcript_base import (
    TRANSCRIPT_SYSTEM,
    build_user_prompt,
    load_transcript,
)


class GPT54TranscriptEvaluator(BaseEvaluator):
    """GPT-5.4 transcript-only baseline."""

    DEPLOYMENT_NAME = "gpt-5.4_2026-03-05"

    def load(self):
        self.deployment_name        = self.DEPLOYMENT_NAME
        # Azure accepts {none, minimal, low, medium, high, xhigh}.
        # Transcript-only is meant as a *cheap, no-thinking* baseline so we
        # default to 'none' (override with GPT_REASONING_EFFORT=medium / high
        # if you want a thinking variant for comparison).
        self.reasoning_effort       = os.environ.get(
            "GPT_REASONING_EFFORT", "none"
        ).strip().lower()
        if self.reasoning_effort in {"", "auto", "default"}:
            self.reasoning_effort = "none"
        # Transcripts can be long → bigger budget for the answer is safe.
        self.max_completion_tokens  = int(os.environ.get(
            "GPT_MAX_COMPLETION_TOKENS", "1024"
        ))
        # Optional cap on transcript characters (None = full transcript).
        max_chars_env = os.environ.get("TRANSCRIPT_MAX_CHARS", "").strip()
        self.transcript_max_chars   = int(max_chars_env) if max_chars_env else None
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
        try:
            transcript = load_transcript(
                video_path,
                keep_timestamps=True,
                max_chars=self.transcript_max_chars,
            )
        except FileNotFoundError as e:
            raise RuntimeError(f"transcript_missing: {e}")

        user_text = build_user_prompt(transcript, prompt)
        messages = [
            {"role": "system", "content": TRANSCRIPT_SYSTEM},
            {"role": "user",   "content": user_text},
        ]
        return self._call_with_fallback(messages)


class GPT51TranscriptEvaluator(GPT54TranscriptEvaluator):
    """GPT-5.1 transcript-only variant (same plumbing, older deployment)."""

    DEPLOYMENT_NAME = "gpt-5.1_2025-11-13"

    def load(self):
        super().load()
        self.deployment_name = self.DEPLOYMENT_NAME
