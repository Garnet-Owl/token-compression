# Changelog

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
