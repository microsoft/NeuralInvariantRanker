# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import random
import warnings
from torch import nn
import numpy as np
import torch
from transformers import is_torch_tpu_available
from transformers.trainer_utils import EvalPrediction
from transformers.file_utils import is_sagemaker_mp_enabled
from transformers.trainer import (
    Trainer,
    OPTIMIZER_NAME,
    SCHEDULER_NAME,
    SCALER_NAME,
    TRAINER_STATE_NAME
)
from typing import Dict, Union, Tuple, Optional, List, Any
from transformers.trainer_pt_utils import reissue_pt_warnings
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    ShardedDDPOption
)
from transformers.utils import logging
import os
import sys

if True:
    project_dir = os.path.abspath(
        os.path.join(os.path.abspath(__file__), "../../.."))
    if project_dir not in sys.path:
        sys.path.append(project_dir)
    from src.ranker import util
    logger = util.get_logger(__file__)


# if is_torch_tpu_available():
#     import torch_xla.core.xla_model as xm

# if is_sagemaker_mp_enabled():
#     import smdistributed.modelparallel.torch as smp

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
    # print(similarity_scores.shape, labels.shape)
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
            
    def _load_rng_state(self, checkpoint):
        # logger.info("Loading the random states!")
        # Load RNG states from `checkpoint`
        if checkpoint is None:
            return
        # local_rank = xm.get_local_ordinal() if is_torch_tpu_available() else self.args.local_rank
        local_rank = self.args.local_rank
        if local_rank != -1:
            rng_file = os.path.join(checkpoint, f"rng_state_{local_rank}.pth")
            if not os.path.isfile(os.path.join(checkpoint, rng_file)):
                logger.info(
                    f"Didn't find an RNG file for process {local_rank}, if you are resuming a training that "
                    "wasn't launched in a distributed fashion, reproducibility is not guaranteed."
                )
                return
        else:
            rng_file = os.path.join(checkpoint, "rng_state.pth")
            if not os.path.isfile(os.path.join(checkpoint, rng_file)):
                logger.info(
                    "Didn't find an RNG file, if you are resuming a training that was launched in a distributed "
                    "fashion, reproducibility is not guaranteed."
                )
                return
        checkpoint_rng_state = torch.load(rng_file)
        random.setstate(checkpoint_rng_state["python"])
        np.random.set_state(checkpoint_rng_state["numpy"])
        torch.random.set_rng_state(checkpoint_rng_state["cpu"])
        if torch.cuda.is_available():
            try:
                if self.args.local_rank != -1:
                    torch.cuda.random.set_rng_state(checkpoint_rng_state["cuda"])
                else:
                    torch.cuda.random.set_rng_state_all(checkpoint_rng_state["cuda"])
            except Exception as ex:
                logger.info(
                    f"""Error encountered while loading the states, you may have used different numbers of GPUs
                    Error Message {ex}
                    """
                )
        # if is_torch_tpu_available():
        #     xm.set_rng_state(checkpoint_rng_state["xla"])

    def save_checkpoint(self):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"
        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        run_dir = self.args.output_dir
        self.store_flos()
        output_dir = os.path.join(run_dir, checkpoint_folder)
        self.save_model(output_dir)
        if self.deepspeed:
            # under zero3 model file itself doesn't get saved since it's bogus! Unless deepspeed
            # config `stage3_gather_fp16_weights_on_model_save` is True
            self.deepspeed.save_checkpoint(output_dir)
        # Save optimizer and scheduler
        if self.sharded_ddp == ShardedDDPOption.SIMPLE:
            self.optimizer.consolidate_state_dict()
        # if is_torch_tpu_available():
        #     xm.rendezvous("saving_optimizer_states")
        #     xm.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))
        #     with warnings.catch_warnings(record=True) as caught_warnings:
        #         xm.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
        #         reissue_pt_warnings(caught_warnings)
        # elif is_sagemaker_mp_enabled():
        #     if smp.dp_rank() == 0:
        #         # Consolidate the state dict on all processed of dp_rank 0
        #         opt_state_dict = self.optimizer.state_dict()
        #         # Save it and the scheduler on the main process
        #         if self.args.should_save:
        #             torch.save(opt_state_dict, os.path.join(output_dir, OPTIMIZER_NAME))
        #             with warnings.catch_warnings(record=True) as caught_warnings:
        #                 torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
        #             reissue_pt_warnings(caught_warnings)
        #             if self.use_amp:
        #                 torch.save(self.scaler.state_dict(), os.path.join(output_dir, SCALER_NAME))
        elif not self.deepspeed:
            # deepspeed.save_checkpoint above saves model/optim/sched
            # torch.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))
            # with warnings.catch_warnings(record=True) as caught_warnings:
            #     torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
            # reissue_pt_warnings(caught_warnings)
            # if self.use_amp:
            #     torch.save(self.scaler.state_dict(), os.path.join(output_dir, SCALER_NAME))
            pass
        self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))
        # Save RNG state in non-distributed training
        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            if self.args.local_rank == -1:
                # In non distributed, we save the global CUDA RNG state (will take care of DataParallel)
                rng_states["cuda"] = torch.cuda.random.get_rng_state_all()
            else:
                rng_states["cuda"] = torch.cuda.random.get_rng_state()
        # if is_torch_tpu_available():
        #     rng_states["xla"] = xm.get_rng_state()
        # A process can arrive here before the process 0 has a chance to save the model,
        # in which case output_dir may not yet exist.
        os.makedirs(output_dir, exist_ok=True)
        # local_rank = xm.get_local_ordinal() if is_torch_tpu_available() else self.args.local_rank
        local_rank = self.args.local_rank
        if local_rank == -1:
            torch.save(rng_states, os.path.join(output_dir, "rng_state.pth"))
        else:
            torch.save(rng_states, os.path.join(output_dir, f"rng_state_{local_rank}.pth"))
        self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)
