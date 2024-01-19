# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from transformers.training_args import TrainingArguments
import os
import sys
if True:
    project_dir = os.path.abspath(
        os.path.join(os.path.abspath(__file__), "../../../..")
    )
    if project_dir not in sys.path:
        sys.path.append(project_dir)
    from src.ranker.codex.data import CrossDataSetForCodex
    from src.ranker.codex.data import CrossLangSearchDataCollatorforCodex


if __name__ == "__main__":
    data_dir = f"{os.environ['HOME']}/REINFOREST/data/atcoder/semantic_match_data"
    data_files = sorted([
        os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".jsonl")
    ])
    print(data_files)
    arguments = TrainingArguments(output_dir="/tmp/output")
    data = CrossDataSetForCodex(
        path=data_dir,
        data_files=data_files,
        name="dataloading-v1-codex",
        cache_dir=os.path.join(data_dir, "cached-codex"),
        num_workers=24,
        training_arguments=arguments,
        load_from_cache=False,
        max_positive_examples=6,
        max_negative_examples=4
    )
    collator = CrossLangSearchDataCollatorforCodex()
    print("=" * 100)

    examples = [data[i] for i in [1, 2, 3, 4, 5]]
    batch = collator(examples)
    for k, v in batch.items():
        print(k, v.shape)
