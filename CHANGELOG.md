# Changelog

## [2025-12-20] - Phase 4: Main Runner API

- Streamlined `app/main_runner/` to 3 files:
  - `config.py`: Config dataclass (model, adapter, API settings)
  - `qa_service.py`: QAService - loads model, compresses context, answers questions
  - `api.py`: FastAPI server with `/ask` endpoint
- Removed redundant files (cli.py, pipeline.py, service.py)
- Run server: `python -m app.main_runner.api`
- POST `/ask` with `{context, question}` → `{answer, compression_ratio}`

## [2025-12-20] - Phase 3: Comprehensive Kaggle Notebook

- Created comprehensive `notebooks/kaggle_finetune.ipynb` with 16 cells
- Each package in its own cell: token_compression, shared/logger, llm_integration, fine_tuning
- Includes all dictionary tokens (① to ⑩ + Ⓐ to Ⓠ) for complete compression support
- Added evaluation cell that tests decompression on 10 samples with accuracy reporting
- Notebook is standalone and runs on Kaggle with GPU (user uploads training_pairs.json)

## [2025-12-20] - Phase 3: Fine-Tuning Setup

- Created `app/fine_tuning/` package with LoRA-based training pipeline
- Added dependencies: peft, trl, datasets, accelerate
- `config.py`: LoRA (r=16, alpha=32) and training hyperparameters
- `dataset.py`: Loads training pairs as `"<compressed> → <original>"` prompts
- `trainer.py`: `DecompressionTrainer` with PEFT/SFTTrainer integration
- `service.py`: CLI entry point with `--test-mode` flag
- Verified: test mode training (3 epochs, 43.7% token accuracy, adapter saved)

## [2025-12-20] - Phase 2: Custom Attention + Training Data Gen

- Created `app/llm_integration/attention.py` with `DecompressionAwareAttention`
- Verified: 24 attention layers replaced, 12 reference token IDs tracked
- Created `app/shared/logger.py` with centralized logging configuration
- Replaced all print statements with logging throughout codebase
- Created `app/training_data_generation/` package:
  - `prompts.py`: LLM prompts for generating diverse repetitive text samples
  - Created `data/raw_texts.txt` with 100 diverse, repetitive text samples
- Updated `app/training_data_generation/pair_generator.py` to support file-based processing
- Generated `data/training_pairs.json` with 100 (original, compressed) training pairs
- Verified compression performance: ratios typically between 1.1x and 2.5x
