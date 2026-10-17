import json
import logging
import os
import random

import numpy as np
import torch
import wandb

from args import HBGLTestingArguments, HBGLTrainingArguments, parse_to_training_arguments
from prepare import get_model_and_tokenizer
from train import HBGLTrainer
from test_model import test as test_main
from utils import TrainingExample, load_and_cache_examples

LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
logger = logging.getLogger(__name__)
logging.basicConfig(level=LOGLEVEL)

def prepare(args: HBGLTrainingArguments):
    """
    Taken from https://github.com/kongds/HBGL/blob/master/run.py

    Removed some process because of deprecated dependencies and simplicity:
    - Distant debugging
    - args.fp16 (including apex)
    """
    prepare_output_dir(args)
    setup_cuda_gpu_and_distributed_training(args)
    setup_logging(args)
    set_seed(args)
    logger.info("Training/evaluation parameters %s", args)
    init_wandb_if_needed(args)

def init_wandb_if_needed(args: HBGLTrainingArguments):
    if (not args.only_test) and args.wandb:
        wandb.init(
            project="HBGL",
            name=args.output_dir.split('/')[-1],
        )
        wandb.define_metric("train/global_step")
        wandb.define_metric("*", step_metric="train/global_step", step_sync=True)

def set_seed(args: HBGLTrainingArguments):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def setup_logging(args: HBGLTrainingArguments):
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                   args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1))

def setup_cuda_gpu_and_distributed_training(args: HBGLTrainingArguments):
    if (args.local_rank != -1) and (not args.no_cuda):
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl')

def prepare_output_dir(args: HBGLTrainingArguments):
    os.makedirs(args.output_dir, exist_ok=True)
    json.dump(args.__dict__, open(os.path.join(
        args.output_dir, 'train_opt.json'), 'w'), sort_keys=True, indent=2)

def test(
        args: HBGLTrainingArguments,
        best_macro_f1_path: str | None,
        best_micro_f1_path: str | None
):
    assert args.test_file is not None
    assert args.add_vocab_file is not None
    
    bout = None
    for i, save_path in enumerate([best_micro_f1_path, best_macro_f1_path]):
        if save_path is None: continue

        out = test_main(HBGLTestingArguments(
            tokenizer_name=args.model_name_or_path,
            input_file=args.test_file,
            split="test",
            model_path=save_path,
            max_seq_length=args.max_source_seq_length,
            max_tgt_length=args.max_target_seq_length,
            batch_size=128,
            cached_features_file=os.path.join(args.output_dir, "cached_features_for_test.pt"),
            add_vocab_file=args.add_vocab_file,
            soft_label_hier_real_with_train_file=(
                args.train_file if args.soft_label_hier_real else None
            )
        ))
        prefix = 'test' + 'micro' if i == 0 else 'macro'
        if args.wandb and out is not None:
            wandb.log({f'{prefix}/macro_f1': out['macro_f1'], f'{prefix}/micro_f1': out['micro_f1']})
            if bout is None or bout['macro_f1'] < out['macro_f1']:
                bout = out

    if args.wandb and bout:
        prefix = 'test'
        wandb.log({f'{prefix}/macro_f1': bout['macro_f1'], f'{prefix}/micro_f1': bout['micro_f1']})

def set_hier_labels_in_model(model, hier_labels):
    def to_multi_hot(label):
        _label = torch.zeros(model.config.vocab_size)
        for i in label:
            _label[i] = 1
        return _label.bool()

    model.hier_labels = [to_multi_hot(i) for i in hier_labels]

def main():
    args = parse_to_training_arguments()
    prepare(args)
    if args.only_test:
        args.wandb = False
        test(args, args.only_test_path, None)
        return

    model, tokenizer, old_vocab_size = get_model_and_tokenizer(args)
    training_features = load_and_cache_examples(
        example_file=args.train_file,
        tokenizer=tokenizer,
        local_rank=args.local_rank,
        cached_features_file=(
            os.path.join(args.output_dir, "cached_features_for_training.pt")
            if args.cached_train_features_file is None
            else args.cached_train_features_file
        ) + "soft_label",
        shuffle=True,
    )
    if args.add_vocab_file:
        assert_correct_target_id(old_vocab_size, training_features)
    trainer = HBGLTrainer(args)
    best_macro_f1_path, best_micro_f1_path = trainer.train(training_features, model, tokenizer)
    if args.test_file:
        test(args, best_macro_f1_path, best_micro_f1_path)

def assert_correct_target_id(
        old_vocab_size: int,
        training_features: list[TrainingExample]
):
    for example in training_features:
        for target_id_list in example.target_ids:
            for target_id in target_id_list:
                assert target_id >= old_vocab_size

if __name__ == "__main__":
    main()
