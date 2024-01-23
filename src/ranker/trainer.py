# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from torch import nn
import numpy as np
import torch
from transformers import EvalPrediction
from transformers import Trainer

from typing import Dict, Union, Tuple, Optional, List, Any
import os
import sys

if True:
    project_dir = os.path.abspath(
        os.path.join(os.path.abspath(__file__), "../../.."))
    if project_dir not in sys.path:
        sys.path.append(project_dir)
    from src.ranker import util
    logger = util.get_logger(__file__)


def _mean(values):
    if isinstance(values, list):
        if len(values) == 0:
            return 0
        elif len(values) == 1:
            return values[0]
        else:
            return np.mean(values).item()       
    else:
        return values 


def compute_metrics(
    eval_prediction: EvalPrediction, in_eval=False
) -> Dict[str, float]:
    similarity_scores = eval_prediction.predictions # (NUM_SAMPLE, D), (1000, 10)
    labels = eval_prediction.label_ids # (NUM_SAMPLE, D)
    num_ex, _ = similarity_scores.shape
    count_at_k = {1: [], 2: [], 5: [], 10: [], 20: [], 50: [], 100: []}
    v_at_k = {1: [], 2: [], 5: [], 10: [], 20: [], 50: [], 100: []}
    first_positive_rank = []
    for exid in range(num_ex):
        similarities = similarity_scores[exid, :]
        mask = labels[exid, :]
        soreted_indices = np.argsort(similarities)[::-1]
        for k in count_at_k.keys():
            taken_indices = soreted_indices[:k]
            count = 0
            for i in taken_indices:
                if mask[i] == 1:
                    count += 1
            if count > 0:
                v_at_k[k].append(1.)
            else:
                v_at_k[k].append(0.)
            count_at_k[k].append(count)
        first_positive_found = False
        for rank, idx in enumerate(soreted_indices):
            if mask[idx] == 1 and not first_positive_found:
                first_positive_rank.append(rank + 1)
                first_positive_found = True
    result = {}
    detailed_res = {}
    for k in v_at_k.keys():
        v = _mean(v_at_k[k])
        result[f'avg_v_at_{k}'] = round(v*100, 4) 
        detailed_res[f'avg_v_at_{k}'] = [x* 100  for x in v_at_k[k]]
        # c = _mean(count_at_k[k])
        # result[f'avg_count_at_{k}'] = round(c*100, 4)
        detailed_res[f'avg_count_at_{k}'] = [x* 100  for x in count_at_k[k]]
    result['avg_first_pos_rank'] = round(_mean(first_positive_rank), 4)
    detailed_res['avg_first_pos_rank'] = first_positive_rank
    if in_eval:
        result['details'] = detailed_res
    return result


class CrossLangCodeSearchTrainer(Trainer):
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.
        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.autocast_smart_context_manager():
                outputs = model(**inputs)
                loss = outputs["loss"]
                positives = outputs["scores"]["positive"]
                positives_labels = torch.ones_like(positives)
                negatives = outputs["scores"]["negative"]
                negative_labels = torch.zeros_like(negatives)
                logits = torch.cat([positives, negatives], dim=-1)
                labels = torch.cat([positives_labels, negative_labels], dim=-1)
                return (loss, logits, labels)
            