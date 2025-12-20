import torch
from peft import LoraConfig as PeftLoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from app.fine_tuning.config import FineTuningConfig
from app.shared.logger import get_logger

logger = get_logger(__name__)


class DecompressionTrainer:
    """Trainer for fine-tuning LLM on compressed text decompression."""

    def __init__(self, config: FineTuningConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None

    def load_model(self):
        logger.info(f"Loading model: {self.config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            dtype=torch.float32,
            attn_implementation="eager",
        )
        self._add_special_tokens()
        self._apply_lora()
        logger.info("Model loaded with LoRA adapters")

    def _add_special_tokens(self):
        special_tokens = ["‹", "›", "①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨", "⑩"]
        new_tokens = [
            t for t in special_tokens if self.tokenizer.convert_tokens_to_ids(t) == self.tokenizer.unk_token_id
        ]
        if new_tokens:
            self.tokenizer.add_tokens(new_tokens)
            self.model.resize_token_embeddings(len(self.tokenizer))
            logger.info(f"Added {len(new_tokens)} special tokens")

    def _apply_lora(self):
        lora_cfg = self.config.lora
        peft_config = PeftLoraConfig(
            r=lora_cfg.r,
            lora_alpha=lora_cfg.lora_alpha,
            target_modules=lora_cfg.target_modules,
            lora_dropout=lora_cfg.lora_dropout,
            bias=lora_cfg.bias,
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

    def train(self, train_dataset: Dataset, eval_dataset: Dataset):
        if self.model is None:
            self.load_model()

        tc = self.config.training
        use_cuda = torch.cuda.is_available()
        sft_config = SFTConfig(
            output_dir=str(self.config.output_dir),
            num_train_epochs=tc.num_epochs,
            per_device_train_batch_size=tc.batch_size,
            gradient_accumulation_steps=tc.gradient_accumulation_steps,
            learning_rate=tc.learning_rate,
            warmup_steps=tc.warmup_steps,
            fp16=use_cuda,
            bf16=False,
            use_cpu=not use_cuda,
            logging_steps=tc.logging_steps,
            save_steps=tc.save_steps,
            eval_strategy="steps",
            eval_steps=tc.save_steps,
            save_total_limit=2,
            dataset_text_field="text",
        )

        trainer = SFTTrainer(
            model=self.model,
            args=sft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
        )

        logger.info("Starting training")
        trainer.train()
        logger.info("Training complete")

        self.model.save_pretrained(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)
        logger.info(f"Saved adapter to {self.config.output_dir}")
