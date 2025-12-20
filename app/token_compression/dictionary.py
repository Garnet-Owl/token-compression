# Dictionary: Common phrases -> single-char tokens
DICTIONARY = {
    # --- Top Frequency 3-grams & 2-grams (Huge Savings) ---
    " of the ": "①",
    " in the ": "②",
    " to the ": "③",
    " on the ": "④",
    " and the ": "⑤",
    " for the ": "⑥",
    " at the ": "⑦",
    " is the ": "⑧",
    " with the ": "⑨",
    " from the ": "⑩",
    " by the ": "⑪",
    " as the ": "⑫",
    # --- Structural/Common Words ---
    "the ": "⑬",
    "and ": "⑭",
    "that ": "⑮",
    "with ": "⑯",
    "from ": "⑰",
    "have ": "⑱",
    "this ": "⑲",
    "will ": "⑳",
    "your ": "Ⓐ",
    "they ": "Ⓑ",
    "been ": "Ⓒ",
    "were ": "Ⓓ",
    "would ": "Ⓔ",
    "could ": "Ⓕ",
    "should ": "Ⓖ",
    "there ": "Ⓗ",
    "their ": "Ⓘ",
    "which ": "Ⓙ",
    "about ": "Ⓚ",
    "other ": "Ⓛ",
    "these ": "Ⓜ",
    "those ": "Ⓝ",
    "being ": "Ⓞ",
    "through ": "Ⓟ",
    "because ": "Ⓠ",
    "between ": "Ⓡ",
    "however ": "Ⓢ",
    "without ": "Ⓣ",
    "another ": "Ⓤ",
    "something ": "Ⓥ",
    "different ": "Ⓦ",
    "important ": "Ⓧ",
    "while ": "Ⓨ",
    "where ": "Ⓩ",
    "every ": "ⓐ",
    "start ": "ⓑ",
    "after ": "ⓒ",
    "before ": "ⓓ",
    "thing ": "ⓔ",
    "place ": "ⓕ",
    "great ": "ⓖ",
    "right ": "ⓗ",
    "again ": "ⓘ",
    "small ": "ⓙ",
    "large ": "ⓚ",
    "found ": "ⓛ",
    "sound ": "ⓜ",
    "under ": "ⓝ",
    "still ": "ⓞ",
    "never ": "ⓟ",
    "house ": "ⓠ",
    "world ": "ⓡ",
    "below ": "ⓢ",
    "asked ": "ⓣ",
    "going ": "ⓤ",
    "until ": "ⓦ",
    "along ": "ⓧ",
    "shall ": "ⓨ",
    "often ": "0",
    "whole ": "1",
    "among ": "2",
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
