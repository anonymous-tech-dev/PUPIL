"""
_abstain_utils.py — Shared abstention detector for the SoF-DPO v2 pipeline.

A "rejected" answer that abstains ("I can't see / the transcript doesn't say
/ not enough information") teaches the DPO objective the WRONG lesson, because
when this kind of negative is preferred-against, the model learns "never
abstain → always confabulate". The benchmark explicitly rewards calibrated
abstention on visual / priority axes, so we audit every generated negative
through this filter and (a) retry generation up to N times with escalating
temperature in the v2 generator, then (b) drop the row in the v2 filter if
the model still couldn't be coaxed off the abstention.

Patterns are intentionally conservative — false positives (treating a real
answer as abstention) only cost a re-roll, but false negatives (missing an
abstention) leak harmful pairs into DPO.
"""
from __future__ import annotations

import re

# Compiled once. Patterns are case-insensitive, DOTALL not needed (we run on
# whitespace-normalised text). Each pattern is anchored to a phrasing that
# is essentially never used by a confident on-topic answer.
_ABSTAIN_PATTERNS = [
    # First-person inability
    r"\bi\s*(?:can(?:not|'?t)|am\s+unable\s+to|do\s*not?\s+have\s+(?:enough\s+)?"
    r"(?:information|context|details|access)|don'?t\s+have\s+(?:enough\s+)?"
    r"(?:information|context|details|access))",
    r"\bi\s+(?:cannot|can'?t)\s+(?:see|tell|determine|identify|know|verify|"
    r"confirm|access|view|watch|hear|listen|provide|infer|read|make\s+out)\b",
    r"\bunable\s+to\s+(?:see|tell|determine|identify|know|verify|confirm|access|"
    r"view|hear|listen|provide|infer|read|answer|make\s+out)\b",
    r"\b(?:it\s+is|it'?s)\s+not\s+possible\s+to\b",
    r"\bno\s+way\s+to\s+(?:tell|determine|know|verify)\b",

    # Source-of-information disclaimers
    r"\bthe\s+(?:video|transcript|speaker|lecture|audio|recording|clip|frames?|"
    r"text|prompt|context|provided\s+information|information\s+provided)\s+"
    r"(?:does(?:n'?t|\s+not)|do(?:n'?t|\s+not))\s+"
    r"(?:mention|show|describe|state|say|provide|specify|contain|include|"
    r"display|indicate|reveal|cover|address|discuss|explain)",
    r"\b(?:does(?:n'?t|\s+not))\s+"
    r"(?:mention|show|describe|state|say|provide|specify|contain|include|"
    r"display|indicate|reveal)\b[^.]{0,40}(?:question|answer|information|"
    r"detail|context|specifics?)",

    # "Without ... I cannot" framings
    r"\bwithout\s+(?:the\s+)?(?:video|transcript|audio|frames?|visual|seeing|"
    r"watching|listening|access)\b[^.]{0,80}(?:cannot|can'?t|unable|impossible|"
    r"hard\s+to|difficult\s+to)",

    # Generic "not enough"
    r"\bnot\s+enough\s+(?:information|context|detail|evidence|content|data)\b",
    r"\binsufficient\s+(?:information|context|detail|evidence|content|data)\b",
    r"\b(?:no|zero)\s+(?:information|context|detail|mention|reference)\b",

    # "Cannot be determined" family
    r"\bcannot\s+be\s+(?:determined|established|inferred|known|verified|"
    r"identified|told|answered)\b",
    r"\b(?:there\s+is|there'?s)\s+no\s+(?:way|indication|mention|reference|"
    r"information)\b",

    # "Is not visible/stated" passive forms
    r"\bis\s+not\s+(?:visible|stated|mentioned|provided|shown|specified|"
    r"described|clear|available|present|given|discussed|explained)\b",
    r"\bare\s+not\s+(?:visible|stated|mentioned|provided|shown|specified|"
    r"described|clear|available|present|given|discussed|explained)\b",

    # Apology / explicit refusal
    r"\b(?:i\s+(?:am\s+sorry|apologi[sz]e|apologi[sz]ed?\s+but)|sorry,)\b"
    r"[^.]{0,80}(?:cannot|can'?t|unable|don'?t\s+have)",

    # "Based on the transcript alone..." style hedge that signals abstention
    # immediately followed by inability
    r"\b(?:based\s+on|from)\s+(?:the\s+)?(?:transcript|text|audio|video)"
    r"\s+(?:alone|only|provided|given)\b[^.]{0,80}"
    r"(?:cannot|can'?t|unable|don'?t\s+have|impossible|hard\s+to)",
]

_ABSTAIN_RE = re.compile("|".join(_ABSTAIN_PATTERNS), re.IGNORECASE)


def is_abstain(text: str) -> bool:
    """Return True if `text` reads as an abstention/refusal-style response."""
    if not text:
        return True
    # Normalise smart quotes & whitespace so the patterns match.
    s = (text.replace("\u2019", "'")
              .replace("\u2018", "'")
              .replace("\u201c", '"')
              .replace("\u201d", '"'))
    s = re.sub(r"\s+", " ", s).strip()
    return bool(_ABSTAIN_RE.search(s))


def abstain_reason(text: str) -> str | None:
    """For diagnostics: return the matched substring, or None."""
    if not text:
        return "empty"
    s = (text.replace("\u2019", "'").replace("\u2018", "'")
              .replace("\u201c", '"').replace("\u201d", '"'))
    s = re.sub(r"\s+", " ", s).strip()
    m = _ABSTAIN_RE.search(s)
    return m.group(0) if m else None


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    samples = [
        ("ABSTAIN", "I cannot see the video, so I am unable to determine the "
                    "color of the slide."),
        ("ABSTAIN", "The transcript does not mention any specific formula."),
        ("ABSTAIN", "Without the video I can't tell what is on the screen."),
        ("ABSTAIN", "Not enough information to answer this question."),
        ("ABSTAIN", "The speaker doesn't say what the value is."),
        ("ABSTAIN", "Based on the transcript alone, I cannot determine the "
                    "specific element shown."),
        ("OK",      "The slide shows BW = 0.35 / t_r as the rule-of-thumb."),
        ("OK",      "Around 12:34 the speaker introduces the diagram and "
                    "labels the three forces acting on the block."),
        ("OK",      "It is a tunnel diode used as a fast trigger element."),
    ]
    for tag, t in samples:
        got = "ABSTAIN" if is_abstain(t) else "OK"
        mark = "OK " if got == tag else "BAD"
        print(f"{mark} expected={tag:7s} got={got:7s} | {t}")
