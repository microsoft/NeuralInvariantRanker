# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
import json
import os
import sys
import numpy as np
import torch.nn as nn
from typing import Type, Dict, Any, List, Callable, Tuple
from transformers.trainer_utils import EvalPrediction
from tqdm import tqdm


if True:
    project_dir = os.path.abspath(
        os.path.join(os.path.abspath(__file__), "../../.."))
    if project_dir not in sys.path:
        sys.path.append(project_dir)
    # from src.ranker.codebert.data import CrossDataSetForCodeBERT
    from src.ranker.codex.data import CrossDataSetForCodex
    from src.ranker.codex.models import Codex
    from src.ranker import util
    logger = util.get_logger(__file__)


def calculate_scores(vector, other_vectors):
    scores = []
    for o in other_vectors:
        scores.append(
            np.dot(vector, o) / (
                np.abs(np.linalg.norm(o, ord=2)) * \
                np.abs(np.linalg.norm(vector, ord=2))
            )
        )
    return np.array(scores)


def batchify(examples, batch_size=32):
    current_idx = 0
    batches = []
    while current_idx < len(examples):
        batches.append(examples[current_idx:current_idx+batch_size])
        current_idx += batch_size
    return batches


class Ranker:
    def __init__(
        self,
        data_class: type,
        model_class: type,
        additional_comp_for_ranker: Dict[str, Any],
    ):
        self.data_class = data_class
        self.model_class = model_class
        self.additional_comp_for_ranker = additional_comp_for_ranker
        assert (
            self.data_class == CrossDataSetForCodex 
        )
        assert "model_name" in self.additional_comp_for_ranker.keys()
        self.cache = {}
        if "cached_embeddings" in self.additional_comp_for_ranker.keys():
            self.cache = self.additional_comp_for_ranker["cached_embeddings"]
            self.additional_comp_for_ranker["cache"] = self.cache
            # logger.info(f"Loaded {len(self.cache.keys())} codes from the cache")
        elif "embedding_path" in self.additional_comp_for_ranker.keys():
            assert os.path.exists(
                self.additional_comp_for_ranker["embedding_path"]
            )
            # logger.info(
            #     f'Loading from  {self.additional_comp_for_ranker["embedding_path"]}'
            # )
            self.cache = json.load(
                open(self.additional_comp_for_ranker["embedding_path"], "r")
            )
            # logger.info(f"Loaded {len(self.cache.keys())} codes from the cache")
            self.additional_comp_for_ranker["cache"] = self.cache
            self.additional_comp_for_ranker.pop("embedding_path")

    def rank_invariants(
        self, 
        model: nn.Module,
        examples: List[Dict[str, Any]],
        _cache = None,
        bar_tqdm=None
    ):
        # if bar_tqdm is None:
        #     bar_tqdm = tqdm
        # local_cache = {}
        if _cache is None:
            cache = {}
        else:
            cache = copy.copy(_cache)
        all_codes = [examples[k]['problem'] for k in range(len(examples))] + \
            [e['code'] for k in range(len(examples)) \
                for e in examples[k]['invariants']] 
        all_codes = list(set(all_codes))
        # logger.info(f'Total Code : {len(all_codes)}')
        not_found_codes = all_codes
        # logger.info(f'Code to be cached : {len(not_found_codes)}')
        batches = batchify(not_found_codes)
        # logger.info(
        #     f"Created {len(batches)} batches for computing the vectors")
        if cache is not None:
            self.additional_comp_for_ranker["cache"] = cache
        # logger.info(self.additional_comp_for_ranker.keys())
        local_cache = {}
        if bar_tqdm is not None:
            batches = bar_tqdm(batches)
        for batch in batches:
            vectors = self.data_class.get_vector(
                model=model, texts=batch, 
                **self.additional_comp_for_ranker
            )
            for t, v in zip(batch, vectors):
                local_cache[t] = v
        # logger.info(f"Inserted {len(local_cache.keys())} codes into the cache")
        sorted_results = []
        if bar_tqdm is None:
            bar = examples
        else:
            bar = bar_tqdm(examples)
        for ex in bar:
            code = ex['problem']
            if code in local_cache:
                code_vector = local_cache[code]
            else:
                code_vector = self.data_class.get_vector(
                    model=model, texts=code, **self.additional_comp_for_ranker
                )
                local_cache[code] = code_vector
            ex['code_vector'] = code_vector
            invariants = ex['invariants']
            for pid, p in enumerate(invariants):
                if p['code'] in local_cache:
                    pv = local_cache[p['code']]
                else:
                    pv = self.data_class.get_vector(
                        model=model, texts=p['code'], 
                        **self.additional_comp_for_ranker
                    )
                cache[p['code']] = pv
                invariants[pid]['code_vector'] = pv
                score = calculate_scores(
                    code_vector, [pv]
                ).tolist()[0]
                invariants[pid]['similarity'] = score
            invariants = sorted(
                invariants, 
                key=lambda x: x['similarity'], 
                reverse=True
            ) 
            ex['invariants'] = invariants
            sorted_results.append(ex)
        return sorted_results

    def rank_invariants_with_classifier(
        self, 
        model: nn.Module,
        examples: List[Dict[str, Any]],
        cache = None,
        bar_tqdm=None
    ):
        if cache is not None:
            local_cache = cache
        elif "cache" in self.additional_comp_for_ranker.keys():
            local_cache = self.additional_comp_for_ranker["cache"]
        else:
            local_cache = {}
        logger.info(f'Cache size : {len(local_cache.keys())}')
        sorted_results = []
        if bar_tqdm is None:
            bar = examples
        else:
            bar = bar_tqdm(examples)
        for ex in bar:
            code = ex['problem']
            invariants = ex['invariants']
            inv_codes = [e['code'] for e in invariants]
            batch_size = 32
            if isinstance(model, Codex):
                code_vector = local_cache[code]
                inv_vectors = [local_cache[e] for e in inv_codes]
                idx = 0
                while idx < len(inv_vectors):
                    batch = inv_vectors[idx:idx+batch_size]
                    scores = model.get_scores(
                        np.array(code_vector), 
                        np.array(batch)
                    )
                    for pid, p in enumerate(invariants[idx:idx+batch_size]):
                        invariants[idx+pid]['similarity'] = scores[pid]
                    idx += batch_size
            else:
                idx = 0
                while idx < len(inv_codes):
                    batch = inv_codes[idx:idx+batch_size]
                    scores = model.get_scores(
                        code, batch
                    )
                    for pid, p in enumerate(invariants[idx:idx+batch_size]):
                        invariants[idx+pid]['similarity'] = scores[pid]
                    idx += batch_size
            invariants = sorted(
                invariants, 
                key=lambda x: x['similarity'], 
                reverse=True
            ) 
            ex['invariants'] = invariants
            sorted_results.append(ex)
        return sorted_results   

    def rank(
        self,
        model: nn.Module,
        examples: List[Dict[str, Any]],
        metric_function: Callable[[EvalPrediction, bool], Dict[str, Any]],
        ignore_no_positives: bool = True,
    ) -> Tuple[Dict[str, float], Dict[str, List[float]], List[Dict[str, Any]]]:
        local_cache = {}
        all_codes = [examples[k]['code'] for k in range(len(examples))] + \
            [e['code'] for k in range(len(examples)) \
                for e in examples[k]['positives']] +\
            [e['code'] for k in range(len(examples)) \
                for e in examples[k]['negatives']]
        all_codes = list(set(all_codes))
        logger.info(f'Total Code to be cached : {len(all_codes)}')
        batches = batchify(all_codes)
        logger.info(
            f"Created {len(batches)} batches for computing the vectors")
        for batch in tqdm(batches):
            vectors = self.data_class.get_vector(
                model=model, texts=batch, 
                **self.additional_comp_for_ranker
            )
            for t, v in zip(batch, vectors):
                local_cache[t] = v
        logger.info(f"Inserted {len(local_cache.keys())} codes into the cache")
        results = {}
        bar = tqdm(examples, desc='avg_v_at_1 = 0.0000\t')
        for ex in bar:
            code = ex['code']
            if code in local_cache:
                code_vector = local_cache[code]
            else:
                code_vector = self.data_class.get_vector(
                    model=model, texts=code, **self.additional_comp_for_ranker
                )
            ex['code_vector'] = code_vector
            full_scores, full_labels = [], []
            for pid, p in enumerate(ex['positives']):
                if p['code'] in local_cache:
                    pv = local_cache[p['code']]
                else:
                    pv = self.data_class.get_vector(
                        model=model, texts=p['code'], 
                        **self.additional_comp_for_ranker
                    )
                local_cache[p['code']] = pv
                ex['positives'][pid]['code_vector'] = pv
                score = calculate_scores(
                    code_vector, [pv]
                ).tolist()[0]
                ex['positives'][pid]['similarity'] = score
                full_scores.append(score)
                full_labels.append(1)
                
            for nid, n in enumerate(ex['negatives']):
                if n['code'] in local_cache:
                    nv = local_cache[n['code']]
                else:
                    nv = self.data_class.get_vector(
                        model=model, texts=n['code'], 
                        **self.additional_comp_for_ranker
                    )
                local_cache[n['code']] = nv
                ex['negatives'][nid]['code_vector'] = nv
                score = calculate_scores(
                    code_vector, [nv]
                ).tolist()[0]
                ex['negatives'][nid]['similarity'] = score
                full_scores.append(score)
                full_labels.append(0)
            if ignore_no_positives and sum(full_labels) == 0:
                continue
            prediction = EvalPrediction(
                predictions=np.array([full_scores]),
                label_ids=np.array([full_labels])
            )
            current_ex_result = metric_function(prediction)
            bar.set_description(
                f'avg_v_at_1 = {round(current_ex_result["avg_v_at_1"], 4)}\t'
            )
            for k in current_ex_result.keys():
                if k not in results.keys():
                    results[k] = []
                results[k].append(current_ex_result[k])

        aggr_result = {
            k: round(np.mean(results[k]), 4) for k in results.keys()
        }
        return aggr_result, results, examples    
            

if __name__ == '__main__':
    v = [2,]