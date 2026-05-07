"""Claude Opus 4.6 via local GH Copilot Server (OpenAI-compatible proxy).

Thin subclass of ClaudeSonnet46CopilotEvaluator that swaps the default model
name to claude-opus-4.6. All other behaviour (frame extraction, retries,
payload construction) is inherited.
"""
import os
from models.claude_sonnet_46 import ClaudeSonnet46CopilotEvaluator


class ClaudeOpus46CopilotEvaluator(ClaudeSonnet46CopilotEvaluator):
    def load(self):
        # Force the Opus default unless caller explicitly overrides via env.
        os.environ.setdefault("CLAUDE_MODEL", "claude-opus-4.6")
        super().load()
