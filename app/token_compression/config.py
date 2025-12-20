from dataclasses import dataclass


@dataclass
class CompressionConfig:
    """Configuration for compression algorithm."""

    window_size: int = 8192
    min_match_length: int = 4
    max_match_length: int = 256
    min_savings: int = 2  # Minimum characters saved to use reference
