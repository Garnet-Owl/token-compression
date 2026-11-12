# Experiments

Each folder is one test. The GPU scripts are written as Kaggle kernels, so they expect Kaggle paths (`/kaggle/input/...`) and internet access.

- [`log_reader/`](log_reader/): fine-tune a small model to read template-compressed logs. This is the main result.
- [`zero_shot_qa/`](zero_shot_qa/): can a model answer from the dictionary + LZ77 format with no training, using a legend in the prompt?
