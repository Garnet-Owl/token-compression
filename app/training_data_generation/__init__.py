# import os
# import re
# from datasets import load_dataset
#
#
# def download_and_format_dataset(dataset_name, subset=None, split="train", num_examples=10000,
#                                 output_file="training_data.txt", text_field="text"):
#     """
#     Downloads a high-quality dataset from Hugging Face and formats it
#     for LZ77 recovery training (one example per line, cleaned).
#     """
#     print(f"Connecting to Hugging Face to fetch {dataset_name} ({subset})...")
#
#     try:
#         # Using wikimedia/wikipedia which is the updated, script-free version
#         # streaming=True is essential for avoiding OOM errors on large datasets
#         ds = load_dataset(dataset_name, subset, split=split, streaming=True)
#
#         count = 0
#         with open(output_file, "w", encoding="utf-8") as f:
#             for entry in ds:
#                 raw_content = entry.get(text_field, "")
#
#                 # Cleaning protocol:
#                 # 1. Remove Wikipedia-specific artifacts (like headers or citations)
#                 # 2. Collapse all whitespace/newlines into single spaces
#                 clean_text = re.sub(r'\s+', ' ', raw_content).strip()
#
#                 # 3. Filtering:
#                 # We want meaty examples for LZ77 distance/length logic.
#                 # Too short = nothing to compress. Too long = context window issues.
#                 if len(clean_text) < 400 or len(clean_text) > 2500:
#                     continue
#
#                 f.write(clean_text + "\n")
#                 count += 1
#
#                 if count % 1000 == 0:
#                     print(f"Progress: {count}/{num_examples} examples saved...")
#
#                 if count >= num_examples:
#                     break
#
#         print(f"Success! Saved {count} high-quality examples to {output_file}")
#
#     except Exception as e:
#         print(f"Error fetching dataset: {e}")
#         print("\nTIP: If you still get a script error, try using 'wikitext' as a backup:")
#         print("dataset_name='wikitext', subset='wikitext-103-raw-v1'")
#
#
# if __name__ == "__main__":
#     # UPDATED: Using wikimedia/wikipedia to avoid the 'Dataset scripts' error.
#     # The '20231101.simple' is a clean, reliable subset.
#     download_and_format_dataset(
#         dataset_name="wikimedia/wikipedia",
#         subset="20231101.simple",
#         output_file="simple_wiki_10k.txt",
#         num_examples=10000
#     )
#
#     # BACKUP OPTION: wikitext is even more stable if Wikipedia has connection/script issues.
#     # download_and_format_dataset(
#     #     dataset_name="wikitext",
#     #     subset="wikitext-103-raw-v1",
#     #     output_file="wikitext_10k.txt",
#     #     num_examples=10000
#     # )
#
