# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import json
import os
import random
import numpy as np
import traceback
import torch
import shutil
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers.training_args import TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
import sys

if True:
    project_dir = os.path.abspath(
        os.path.join(os.path.abspath(__file__), "../../.."))
    if project_dir not in sys.path:
        sys.path.append(project_dir)
    from src.ranker.codex.data import (
        RankerDataSetForCodex,
        RankerDataCollatorforCodex
    )
    from src.ranker.codex.models import (
        CodexBasedModel, CodexBasedClassificationModel
    )
    from src.ranker.trainer import CrossLangCodeSearchTrainer, compute_metrics
    from src.ranker.atomic_code_util import DelayedKeyboardInterrupt
    from src.ranker.ranker import Ranker
    from src.ranker import util
    logger = util.get_logger(__file__)


def set_seeds(seed):
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)


def num_parameters(model):
    model_parameters = model.parameters()
    return sum([np.prod(p.size()) for p in model_parameters])


def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--exp_name', help='Name of the experiment', default='exp'
    )
    parser.add_argument(
        "--training_config", type=str,
        help="Path of the training configuration file", required=True
    )
    parser.add_argument(
        "--data_path", type=str,
        help="Base Directory of processed data", required=True
    )
    parser.add_argument(
        "--output_dir", type=str,
        help="Path of the output directory",
        required=True
    )
    parser.add_argument(
        "--initial_model",
        type=str,
        default='codebert'
    )
    parser.add_argument(
        "--ckpt_path_from_other_exp",
        help='Checkpoint to be loaded from other experiment.'
        'Should be a folder containing pytorch_model.bin file.',
        default=None
    )
    parser.add_argument(
        "--workers", help="Number of worker CPU", type=int, default=20
    )
    parser.add_argument(
        "--data_cache_path", type=str,
        help="Caching Directory of processed data", default=None
    )
    parser.add_argument(
        "--do_not_reload_from_checkpoint", action="store_true",
        help="Flag to forcefully stop reloading from the checkpoint"
    )
    parser.add_argument("--seed", type=int, default=5000)
    parser.add_argument(
        "--overwrite_cache",
        help='Overwrite the cache dataset directory, if such exists', action='store_true'
    )
    parser.add_argument(
        "--local_rank", help="The local rank in distributed training", type=int,
        default=-1
    )
    parser.add_argument(
        '--max_positive_examples', default=5, type=int
    )
    parser.add_argument(
        '--max_negative_examples', default=5, type=int
    )
    parser.add_argument(
        '--codex_model', help='Name of the Model to use for Code Exp',
        default='babbage-code-search-text'
    )
    parser.add_argument(
        '--alpha', type=float, default=0.1
    )
    parser.add_argument(
        '--do_train', action='store_true'
    )
    parser.add_argument(
        '--do_rank', action='store_true'
    )
    parser.add_argument(
        '--rank_result_path', type=str,
        help='Path to store the ranked result'
    )
    parser.add_argument('--raw_data', type=str, default=None)
    parser.add_argument('--embedding_path', type=str, default=None)
    parser.add_argument('--no_train_rank', action='store_true')
    parser.add_argument('--use_classification_model', action='store_true')
    parser.add_argument('--use_multi_class_classification', action='store_true')
    args = parser.parse_args()
    if args.data_cache_path is not None:
        os.makedirs(args.data_cache_path, exist_ok=True)
    return args


def save_best_validation_ckpt(logger, output_dir, training_args, trainer):
    logger.info("#" * 150)
    logger.info("#" * 150)
    logger.info("Saving best model")
    logger.info(trainer.state.best_metric)
    logger.info("#" * 150)
    logger.info("#" * 150)
    best_validation_model_path = os.path.join(
        output_dir, f'checkpoint-best-{training_args.metric_for_best_model}'
    )
    os.makedirs(best_validation_model_path, exist_ok=True)
    logger.info(f"Saving best model to {best_validation_model_path}")
    if os.path.exists(best_validation_model_path):
        shutil.rmtree(best_validation_model_path)
    shutil.copytree(
        trainer.state.best_model_checkpoint,
        best_validation_model_path
    )
    # shutil.rmtree(trainer.state.best_model_checkpoint)


def load_model(logger, model, ckpt_under_check, dont_load=False):
    if not dont_load:
        ckpt_file = os.path.join(ckpt_under_check, 'pytorch_model.bin')
        logger.info(f'Loading model from {ckpt_file}')
        if not os.path.exists(ckpt_file):
            logger.info(f'Model file does not exists. Please train first!')
            exit()
        with open(ckpt_file, 'rb') as fp:
            model.load_state_dict(torch.load(fp))


if __name__ == '__main__':
    args = parse_command_line_args()
    set_seeds(args.seed)
    logger.info('=' * 50)
    logger.info('=' * 50)
    logger.info(args)
    logger.info('=' * 50)
    logger.info('=' * 50)
    # Note: We need this to come earlier for datasets preparation
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    training_argument_dict = json.load(open(args.training_config))
    training_argument_dict["output_dir"] = output_dir
    training_args = TrainingArguments(**training_argument_dict)
    training_args.dataloader_num_workers = args.workers
    if args.local_rank != -1:
        training_args.local_rank = args.local_rank
    cached_embeddings = None
    if args.initial_model == 'codex':
        DATASET = RankerDataSetForCodex
        DATA_COLLATOR = RankerDataCollatorforCodex
        hidden_dim = RankerDataSetForCodex.get_dimension(args.codex_model)
        if args.use_classification_model:
            model  = CodexBasedClassificationModel(
                hidden_dim=hidden_dim,
                model_name=args.codex_model,
                use_binary=not args.use_multi_class_classification,
            )
        else:
            model = CodexBasedModel(
                hidden_dim=hidden_dim,
                model_name=args.codex_model,
                alpha=args.alpha
            )
        tokenizer = None
        model_specific_arguments_for_ranker = {
            "model_name": args.codex_model,
            "no_train_rank": args.no_train_rank,
        }
        if args.embedding_path is not None:
            cached_embeddings = json.load(open(args.embedding_path))
            model_specific_arguments_for_ranker[
                "cached_embeddings"
            ] = cached_embeddings
    else:
        raise NotImplementedError(f"Unknown initial model {args.initial_model}")
    # logger.info(model)
    data_dir = args.data_path
    logger.info(data_dir)
    data_dir = os.path.abspath(data_dir.rstrip("/"))

    if args.do_train:
        if args.initial_model == 'codex':
            assert (
                args.raw_data is not None or args.embedding_path is not None
            ), "Either raw_data or embedding_path should be provided"
        if args.ckpt_path_from_other_exp is not None:
            logger.info(
                f'Loading from from another experiment checkpoint :' +
                f' {args.ckpt_path_from_other_exp}'
            )
            ckpt_file = os.path.join(
                args.ckpt_path_from_other_exp, 'pytorch_model.bin')
            if not os.path.exists(ckpt_file):
                logger.info(
                    f'Model file does not exists. exiting'
                )
                exit()
            with open(ckpt_file, 'rb') as fp:
                try:
                    model.load_state_dict(torch.load(fp))
                except Exception as e:
                    logger.info(e)
                    logger.info(
                        f'The model present in {ckpt_file} does not match current experimental model'
                        'Please change ckpt_path_from_other_exp argument to point to corrent model'
                        'Or remove this argument to fresh start training a model'
                    )
                    exit()
        train_data_file = [os.path.join(data_dir, 'train.jsonl')]
        assert all([os.path.exists(f) for f in train_data_file])
        train_dataset = DATASET(
            path=data_dir,
            data_files=train_data_file,
            name=args.exp_name + "-train",
            tokenizer=tokenizer,
            cache_dir=os.path.join(
                    args.data_cache_path if (
                        args.data_cache_path is not None
                    ) else data_dir,
                "train-cached"
            ),
            num_workers=args.workers,
            training_arguments=training_args,
            load_from_cache=not args.overwrite_cache,
            max_positive_examples=args.max_positive_examples,
            max_negative_examples=args.max_negative_examples,
            codex_model=args.codex_model if args.initial_model == 'codex' else None,
            raw_data=args.raw_data,
            embedding_path=args.embedding_path,
            cached_embeddings=cached_embeddings,
        )
        eval_data_file = [os.path.join(data_dir, 'valid.jsonl')]
        eval_dataset = DATASET(
            path=data_dir,
            data_files=eval_data_file,
            name=args.exp_name + "-eval",
            tokenizer=tokenizer,
            cache_dir=os.path.join(
                    args.data_cache_path if (
                        args.data_cache_path is not None
                    ) else data_dir,
                "eval-cached"
            ),
            num_workers=args.workers,
            training_arguments=training_args,
            load_from_cache=True,
            max_positive_examples=max(args.max_positive_examples, 1),
            max_negative_examples=max(args.max_negative_examples, 1),
            codex_model=args.codex_model if args.initial_model == 'codex' else None,
            raw_data=args.raw_data,
            embedding_path=args.embedding_path,
            cached_embeddings=cached_embeddings,
        )
        # logger.info(train_dataset.max_positive_examples, train_dataset.max_negative_examples)
        # logger.info(eval_dataset.max_positive_examples, eval_dataset.max_negative_examples)
        trainer = CrossLangCodeSearchTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=DATA_COLLATOR(),
            compute_metrics=compute_metrics
        )
        try:
            if args.do_not_reload_from_checkpoint:
                trainer.train()
            else:
                last_checkpoint = get_last_checkpoint(output_dir)
                try:
                    if last_checkpoint is not None:
                        ckpt_number = last_checkpoint.split("-")[-1]
                        try:
                            ckpt_number = int(ckpt_number)
                        except ValueError:
                            pass
                    else:
                        ckpt_number = last_checkpoint
                    if not isinstance(ckpt_number, int):
                        logger.info("Did not find a valid checkpoint")
                        logger.info("Starting from scratch")
                        trainer.train()
                    else:
                        trainer.train(resume_from_checkpoint=last_checkpoint)
                except KeyboardInterrupt:
                    raise KeyboardInterrupt
                except Exception as ex:
                    traceback.print_exc()
                    logger.info(
                        f"Found an exception {ex} of type {type(ex)}. "
                        "Carefully inspect the stacktrace")
                    exit(1)
            if trainer.state.best_model_checkpoint is not None:
                save_best_validation_ckpt(
                    logger, output_dir, training_args, trainer)
        except KeyboardInterrupt:
            with DelayedKeyboardInterrupt():
                logger.info("*" * 50)
                logger.info("*" * 20, "CAUTION", "*" * 20)
                logger.info("Keyboard Interrupt encountered!!")
                logger.info("Saving the checkpoint in ",
                            trainer.state.global_step)
                trainer.save_checkpoint()
                logger.info("Checkpoint Saved!")
                if trainer.state.best_model_checkpoint is not None:
                    save_best_validation_ckpt(
                        logger, output_dir, training_args, trainer)
                logger.info("*" * 70, "CAUTION", "*" * 70)

