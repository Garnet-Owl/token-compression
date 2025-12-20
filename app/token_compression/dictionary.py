# Dictionary: Common phrases -> single-char tokens
DICTIONARY = {
    # High-frequency words
    "the ": "①",
    "and ": "②",
    "that ": "③",
    "with ": "④",
    "from ": "⑤",
    "have ": "⑥",
    "this ": "⑦",
    "will ": "⑧",
    "your ": "⑨",
    "they ": "⑩",
    "been ": "⑪",
    "were ": "⑫",
    "would ": "⑬",
    "could ": "⑭",
    "should ": "⑮",
    "there ": "Ⓐ",
    "their ": "Ⓑ",
    "which ": "Ⓒ",
    "about ": "Ⓓ",
    "other ": "Ⓔ",
    "these ": "Ⓕ",
    "those ": "Ⓖ",
    "being ": "Ⓗ",
    "through ": "Ⓘ",
    "because ": "Ⓙ",
    "between ": "Ⓚ",
    "however ": "Ⓛ",
    "without ": "Ⓜ",
    "another ": "Ⓝ",
    "something ": "Ⓞ",
    "different ": "Ⓟ",
    "important ": "Ⓠ",
}

REVERSE_DICTIONARY = {v: k for k, v in DICTIONARY.items()}

# Base62 characters for compact encoding
BASE62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def to_base62(n: int) -> str:
    """Convert integer to base62 string."""
    if n == 0:
        return "0"
    result = ""
    while n:
        result = BASE62[n % 62] + result
        n //= 62
    return result


def from_base62(s: str) -> int:
    """Convert base62 string to integer."""
    result = 0
    for c in s:
        result = result * 62 + BASE62.index(c)
    return result
