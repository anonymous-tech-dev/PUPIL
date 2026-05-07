"""
InternVL3.5-8B evaluator.

Identical pipeline to InternVL3-8B (see `intern_3_vl.py`) — only the
checkpoint changes. We subclass `InternVL3Evaluator` and just override
`MODEL_ID` so any future fixes propagate to both variants automatically.

Reference: open-compass/VLMEvalKit
  - vlmeval/config.py (InternVL3_5-8B entry, version="V2.0")
"""

from models.intern_3_vl import InternVL3Evaluator


class InternVL35Evaluator(InternVL3Evaluator):
    MODEL_ID = "OpenGVLab/InternVL3_5-8B"
