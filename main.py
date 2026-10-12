import json
import logging
import os
import random

import numpy as np
import torch
from transformers import BertTokenizer
import wandb

from args import HBGLTestingArguments, HBGLTrainingArguments, parse_to_training_arguments
from prepare import get_model_and_tokenizer
from train import train
from test import main as test_main
from utils import load_and_cache_examples

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
    os.makedirs(args.output_dir, exist_ok=True)
    json.dump(args.__dict__, open(os.path.join(
        args.output_dir, 'train_opt.json'), 'w'), sort_keys=True, indent=2)

    # Setup CUDA, GPU & distributed training
    if (args.local_rank != -1) and (not args.no_cuda):
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl')

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                   args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1))

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    logger.info("Training/evaluation parameters %s", args)

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
            do_lower_case=True,
            model_path=save_path,
            max_seq_length=(
                args.max_source_seq_length + args.max_target_seq_length
                if args.label_cpt_decodewithpos
                else args.max_source_seq_length
            ),
            max_tgt_length=args.max_target_seq_length,
            batch_size=128,
            beam_size=1,
            length_penalty=0,
            forbid_duplicate_ngrams=True,
            mode="s2s",
            forbid_ignore_word='"."',
            cached_features_file=os.path.join(args.output_dir, "cached_features_for_test.pt"),
            add_vocab_file=args.add_vocab_file,
            softmax_label_only=args.softmax_label_only,
            soft_label=args.soft_label,
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

    init_wandb_if_active(args)
    model, tokenizer, old_vocab_size = prepare_model_and_tokenizer(args)
    set_args_cached_train_features_file(args)
    training_features = prepare_training_features(args, tokenizer, old_vocab_size)
    best_macro_f1_path, best_micro_f1_path = train(args, training_features, model, tokenizer)
    if args.test_file:
        test(args, best_macro_f1_path, best_micro_f1_path)

def prepare_training_features(
        args: HBGLTrainingArguments,
        tokenizer: BertTokenizer,
        old_vocab_size: int
):
    training_features = load_and_cache_examples(
        example_file=args.train_file, tokenizer=tokenizer, local_rank=args.local_rank,
        cached_features_file=args.cached_train_features_file, shuffle=True,
        lmdb_cache=args.lmdb_cache, lmdb_dtype=args.lmdb_dtype,
        soft_label=args.soft_label,
    )

    if args.add_vocab_file:
        for example in training_features:
            for target_id_list in example.target_ids:
                if args.soft_label:
                    for target_id in target_id_list:
                        assert target_id >= old_vocab_size
    return training_features

def prepare_model_and_tokenizer(args: HBGLTrainingArguments):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
        # Make sure only the first process in distributed training will download model & vocab
    # Load pretrained model and tokenizer
    model, tokenizer, old_vocab_size = get_model_and_tokenizer(args)

    if args.local_rank == 0:
        torch.distributed.barrier()
        # Make sure only the first process in distributed training will download model & vocab

    return model,tokenizer,old_vocab_size

def set_args_cached_train_features_file(args: HBGLTrainingArguments):
    if args.cached_train_features_file is None:
        if not args.lmdb_cache:
            args.cached_train_features_file = os.path.join(args.output_dir, "cached_features_for_training.pt")
        else:
            args.cached_train_features_file = os.path.join(args.output_dir, "cached_features_for_training_lmdb")

    if args.soft_label:
        args.cached_train_features_file += 'soft_label'

def init_wandb_if_active(args: HBGLTrainingArguments):
    if args.wandb:
        wandb.init(
            project="HBGL",
            name=args.output_dir.split('/')[-1],
        )
        wandb.define_metric("train/global_step")
        wandb.define_metric("*", step_metric="train/global_step", step_sync=True)


if __name__ == "__main__":
    main()
