# Zero-shot question answering

Tests whether a model can answer questions from the dictionary + LZ77 format with no training, given a legend that explains the format. The same questions are asked over the original text, the text with stopwords removed, and the compressed text.

- `run_qwen35.py`: the test script. It uses Qwen3.5-4B and a short-answer prompt.
- `results_qwen35.json`: its results.

Tasks: 200 SQuAD validation questions, and 100 synthetic log lookups with 40 lines each.
