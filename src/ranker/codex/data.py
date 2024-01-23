# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
from dataclasses import dataclass
from torch.utils.data import Dataset as TorchDS
from datasets import load_dataset
import openai
import time
from typing import List, Dict, Any, Optional, Union
from transformers.training_args import TrainingArguments
import torch
from transformers.data.data_collator import DefaultDataCollator
from tqdm import tqdm

import os
import sys
import json

if True:
    project_dir = os.path.abspath(
        os.path.join(os.path.abspath(__file__), "../../../.."))
    if project_dir not in sys.path:
        sys.path.append(project_dir)
    from src.ranker.codex.models import CodexBasedModel
    from src.ranker.util import get_logger
    logger = get_logger(__file__)


class RankerDataSetForCodex(TorchDS):
    @classmethod
    def get_dimension(cls, model_name):
        if model_name == 'davinci-similarity':
            return 12288
        elif model_name == 'text-embedding-ada-002':
            return 1536
        else:
            raise Exception(f"Unknown model name {model_name}")

    def __init__(
        self,
        path: str,
        data_files: List[str],
        name: str,
        training_arguments: TrainingArguments,
        cache_dir: Optional[str] = None,
        num_workers: Optional[int] = 16,
        load_from_cache: Optional[bool] = True,
        codex_model: Optional[str] = 'text-embedding-ada-002',
        max_positive_examples: Optional[int] = 3,
        max_negative_examples: Optional[int] = 3,
        *args, **kwargs
    ):
        self.training_args = training_arguments
        self.codex_model = codex_model
        self.max_positive_examples = max_positive_examples
        self.max_negative_examples = max_negative_examples
        if "cached_embeddings" in kwargs and kwargs["cached_embeddings"] is not None:
            cached_embeddings = kwargs["cached_embeddings"]
        elif "embedding_path" in kwargs:
            logger.info(f'Loading from cache {kwargs["embedding_path"]}')
            cached_embeddings = json.load(open(kwargs["embedding_path"]))
            logger.info(f'Loaded {len(cached_embeddings)} embeddings')
        else:
            raise Exception("Initial embeddings not provided")
        self.data = load_dataset(
            path=path, data_dir=path, data_files=data_files, split="train", cache_dir=cache_dir, name=name
        )
        columns = self.data.column_names
        self.count = 0

        def find_embedding(text):
            if text not in cached_embeddings.keys():
                raise Exception(f"Embedding not found for {text}")
            return cached_embeddings[text]
        
        def prepare_features(examples):
            # print(self.count, flush=True)
            self.count += 1
            inputs = examples["code"]
            input_vector = torch.Tensor(
                find_embedding(inputs)
            )
            # Process the positive vectors - upto the max_positives
            positives = examples['positives']
            positives = sorted(positives, key=lambda x:x['score'], reverse=True)
            positive_codes = [p['code'] for p in positives]
            positive_scores = [p['score'] if p['score'] != -1 else 0. for p in positives]
            assert len(positive_codes) == len(positive_scores)
            if len(positive_codes) >= max_positive_examples:
                positive_codes = positive_codes[:max_positive_examples]
                positive_scores = positive_scores[:max_positive_examples]
                positive_vectors = torch.Tensor(
                    [find_embedding(p) for p in positive_codes]
                )
            elif len(positive_codes) < max_positive_examples:
                positive_vectors = [
                    find_embedding(p) for p in positive_codes
                ]
                if len(positive_vectors) > 0:
                    pdim = len(positive_vectors[0])
                else:
                    pdim = len(find_embedding(" "))
                positive_vectors.extend(
                    [[0.] * pdim] * (max_positive_examples -
                                     len(positive_codes))
                )
                positive_vectors = torch.Tensor(positive_vectors)
                positive_scores.extend(
                    [0.] * (max_positive_examples - len(positive_codes))
                )
            positive_scores = torch.Tensor(positive_scores)
            # Now Process the negative vectors - upto the max_negatives
            negatives = examples['negatives']
            negatives = sorted(negatives, key=lambda x:x['score'], reverse=True)
            negative_codes = [p['code'] for p in negatives]
            negative_scores = [p['score'] if p['score'] != -1 else 0. for p in negatives]
            assert len(negative_codes) == len(negative_scores)
            if len(negative_codes) >= max_negative_examples:
                negative_codes = negative_codes[:max_negative_examples]
                negative_scores = negative_scores[:max_negative_examples]
                negative_vectors = torch.Tensor(
                    [find_embedding(p) for p in negative_codes]
                )
            elif len(negative_codes) < max_negative_examples:
                negative_vectors = [
                    find_embedding(p) for p in negative_codes
                ]
                if len(negative_vectors) > 0:
                    ndim = len(negative_vectors[0])
                else:
                    ndim = len(find_embedding(" "))
                negative_vectors.extend(
                    [[0.] * ndim] * (max_negative_examples -
                                     len(negative_codes))
                )
                negative_vectors = torch.Tensor(negative_vectors)
                negative_scores.extend(
                    [0.] * (max_negative_examples - len(negative_codes))
                )
            negative_scores = torch.Tensor(negative_scores)
            return {
                'input_vector': input_vector,
                'positive_vectors': positive_vectors,
                'negative_vectors': negative_vectors,
                'positive_semantic_match_scores': positive_scores,
                'negative_semantic_match_scores': negative_scores,
            }
        with self.training_args.main_process_first(desc=f"dataset map pre-processing {name}"):
            self.data = self.data.map(
                prepare_features,
                batched=False,  # Do not do batched processing, it will not work
                num_proc=1,
                remove_columns=columns,
                load_from_cache_file=load_from_cache,
                desc=f"dataset map pre-processing {name}"
            )

    @classmethod
    def get_vector(
        cls, model_name: str, model: CodexBasedModel, texts: Union[str, List[str]],
        cache: Dict[str, List[float]], no_train_rank: bool = False,
        **kwargs
    ):
        batched = True
        if isinstance(texts, str):
            batched = False
            texts = [texts]
        embeddings = []
        for t in texts:
            if t in cache.keys():
                embeddings.append(cache[t])
            else:
                raise Exception(f"Embedding not found for {t}")
        device = next(model.parameters()).device
        embeddings = torch.Tensor(embeddings)
        embeddings = embeddings.to(device)
        if no_train_rank:
            vector = embeddings
        else:
            # logger.info("Checking for the vector from the model")
            vector = model.get_vector(
                input_vector=embeddings
            )
        if not batched:
            vector = vector.squeeze(0)
        return vector.cpu().numpy().tolist()
        
    def get_vector_from_dataset(
        self,
        model: CodexBasedModel,
        texts: Union[str, List[str]]
    ):
        return RankerDataSetForCodex.get_vector(
            model_name=self.codex_model,
            model=model,
            texts=texts
        )

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


@dataclass
class RankerDataCollatorforCodex(DefaultDataCollator):
    def __call__(
        self,
        features: List[Dict[str, Any]],
        return_tensors=None
    ) -> Dict[str, Any]:
        batch = {}
        first = features[0]
        for k, v in first.items():
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            else:
                try:
                    batch[k] = torch.tensor([f[k] for f in features])
                except Exception as e:
                    print(k)
                    for f in features:
                        print(torch.tensor(f[k]).shape)
                    exit()
        return batch
