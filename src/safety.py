import re
BAD = [
    r"\b(kill yourself|kms)\b",
]
def guard(text: str) -> str:
    out = text
    for pat in BAD:
        out = re.sub(pat, "[removed]", out, flags=re.I)
    return out
