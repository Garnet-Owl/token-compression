# Changelog

## [2025-12-20] - Phase 2: Custom Attention + Training Data Gen

- Created `app/llm_integration/attention.py` with `DecompressionAwareAttention`
- Verified: 24 attention layers replaced, 12 reference token IDs tracked
- Created `app/shared/logger.py` with centralized logging configuration
- Replaced all print statements with logging throughout codebase
- Created `app/training_data_generation/` package:
  - `prompts.py`: LLM prompts for generating diverse repetitive text samples
  - `pair_generator.py`: Processes raw text → (original, compressed) pairs
