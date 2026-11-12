# Zero-shot QA test

Answers questions from the original, stopword-stripped and compressed context. The compressed version comes with a legend that explains the format.

- `run_qwen35.py` with `results_qwen35.json`: Qwen3.5-4B, short-answer prompt. This is the run in the main README.
- `run_qwen25.py` with `results_qwen25.json`: Qwen2.5-1.5B, looser prompt. Kept for reference.

Tasks: 200 SQuAD validation questions, and 100 synthetic log lookups (40 lines each).
