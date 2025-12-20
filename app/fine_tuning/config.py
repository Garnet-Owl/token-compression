from dataclasses import dataclass, field
from typing import List
from pathlib import Path


@dataclass
class LoraConfig:
    r: int = 16
    lora_alpha: int = 32
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    lora_dropout: float = 0.1
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainingConfig:
    batch_size: int = 4
    num_epochs: int = 3
    learning_rate: float = 2e-4
    warmup_steps: int = 10
    max_seq_length: int = 512
    gradient_accumulation_steps: int = 4
    fp16: bool = True
    logging_steps: int = 10
    save_steps: int = 50


@dataclass
class FineTuningConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B"
    data_path: Path = field(default_factory=lambda: Path("data/training_pairs.json"))
    output_dir: Path = field(default_factory=lambda: Path("models/lora_adapter"))
    lora: LoraConfig = field(default_factory=LoraConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    test_mode: bool = False
    test_samples: int = 5
