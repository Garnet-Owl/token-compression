from typing import Tuple
from app.token_compression.dictionary import REVERSE_DICTIONARY, from_base62


class Decompressor:
    """
    Decompression logic to expand compressed text.
    """

    @staticmethod
    def _parse_ref(ref: str) -> Tuple[int, int]:
        """Parse reference to (distance, length)."""
        inner = ref[1:-1]  # Remove ‹ and ›
        d_str, l_str = inner.split(":")
        return from_base62(d_str), from_base62(l_str)

    @staticmethod
    def _reverse_dictionary(text: str) -> str:
        """Expand dictionary tokens back to phrases."""
        result = text
        for token, phrase in REVERSE_DICTIONARY.items():
            result = result.replace(token, phrase)
        return result

    @staticmethod
    def _expand_overlap(current: str, distance: int, length: int) -> str:
        """Handle overlapping copy for LZ77."""
        start = len(current) - distance
        if start < 0:
            return ""

        copied = ""
        for j in range(length):
            src_idx = start + j
            if src_idx < len(current):
                copied += current[src_idx]
            else:
                # Overlapping: copy from what we just added
                copied += copied[j - distance]
        return copied

    def decompress(self, compressed: str) -> str:
        """
        Decompress text by expanding references.
        """
        # Phase 1: Expand LZ77 references
        output = []
        i = 0

        while i < len(compressed):
            if compressed[i] == "‹":
                end = compressed.find("›", i)
                if end > i:
                    ref = compressed[i : end + 1]
                    try:
                        distance, length = self._parse_ref(ref)
                        current = "".join(output)
                        copied = self._expand_overlap(current, distance, length)
                        if copied:
                            output.append(copied)
                            i = end + 1
                            continue
                    except ValueError:
                        pass  # Valid character that looks like start of ref but isn't

            output.append(compressed[i])
            i += 1

        # Phase 2: Expand dictionary tokens
        return self._reverse_dictionary("".join(output))
