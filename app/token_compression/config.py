from dataclasses import dataclass


@dataclass
class CompressionConfig:
    """Configuration for compression algorithm."""

    window_size: int = 8192
    min_match_length: int = 3
    max_match_length: int = 400
    min_savings: int = 1  # Minimum characters saved to use reference
