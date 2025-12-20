from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    """Configuration for the compressed Q&A service."""

    base_model: str = "Qwen/Qwen2.5-0.5B"
    adapter_path: Path = Path("models/lora_adapter")
    max_tokens: int = 512
    api_host: str = "0.0.0.0"
    api_port: int = 8000
