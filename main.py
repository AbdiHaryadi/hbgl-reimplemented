import argparse
import json
import logging
import os
import random

import numpy as np
import torch
import wandb

from prepare import get_model_and_tokenizer
from train import train
from test import main as test_main
from utils import load_and_cache_examples

LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
logger = logging.getLogger(__name__)
logging.basicConfig(level=LOGLEVEL)

def get_args():
    """
    Taken from https://github.com/kongds/HBGL/blob/master/run.py

    Removed some parameters because deprecated in newer dependencies:
    - --server_ip
    - --server_port
    - --model_type
    """

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--train_file", default=None, type=str, required=True,
                        help="Training data (json format) for training. Keys: source and target")
    parser.add_argument("--valid_file", default=None, type=str, required=True,
                        help="Training data (json format) for training. Keys: source and target")
    parser.add_argument("--test_file", default=None, type=str,
                        help="Training data (json format) for training. Keys: source and target")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list:")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--log_dir", default=None, type=str,
                        help="The output directory where the log will be written.")

    ## Other parameters
    parser.add_argument("--config_name", default=None, type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default=None, type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default=None, type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument("--max_source_seq_length", default=464, type=int,
                        help="The maximum total source sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--max_target_seq_length", default=48, type=int,
                        help="The maximum total target sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")

    parser.add_argument("--cached_train_features_file", default=None, type=str,
                        help="Cached training features file")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--label_smoothing", default=0.1, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_training_steps", default=-1, type=int,
                        help="set total number of training steps to perform")
    parser.add_argument("--num_training_epochs", default=10, type=int,
                        help="set total number of training epochs to perform (--num_training_steps has higher priority)")
    parser.add_argument("--num_warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument("--random_prob", default=0.1, type=float,
                        help="prob to random replace a masked token")
    parser.add_argument("--keep_prob", default=0.1, type=float,
                        help="prob to keep no change for a masked token")
    parser.add_argument("--fix_word_embedding", action='store_true',
                        help="Set word embedding no grad when finetuning.")

    parser.add_argument('--logging_steps', type=int, default=500,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=1500,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")

    parser.add_argument('--source_mask_prob', type=float, default=-1.0,
                        help="Probability to mask source sequence in fine-tuning")
    parser.add_argument('--target_mask_prob', type=float, default=0.5,
                        help="Probability to mask target sequence in fine-tuning")
    parser.add_argument('--num_max_mask_token', type=int, default=0,
                        help="The number of the max masked tokens in target sequence")
    parser.add_argument('--mask_way', type=str, default='v2',
                        help="Fine-tuning method (v0: position shift, v1: masked LM, v2: pseudo-masking)")
    parser.add_argument("--lmdb_cache", action='store_true',
                        help="Use LMDB to cache training features")
    parser.add_argument("--lmdb_dtype", type=str, default='h',
                        help="Data type for cached data type for LMDB")

    parser.add_argument("--add_vocab_file", type=str, default=None)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--softmax_label_only', action='store_true')

    parser.add_argument('--soft_label', action='store_true')
    parser.add_argument('--soft_label_hier_real', action='store_true')

    parser.add_argument('--one_by_one_label_init_map', type=str, default=None)
    parser.add_argument('--label_cpt', type=str, default=None)
    parser.add_argument('--label_cpt_lr', type=float, default=1e-3)
    parser.add_argument('--label_cpt_steps', type=int, default=500)
    parser.add_argument('--label_cpt_bsz', type=int, default=32)
    parser.add_argument('--label_cpt_not_incr_mask_ratio', action='store_true')
    parser.add_argument('--label_cpt_use_bce', action='store_true')

    parser.add_argument('--label_cpt_decodewithpos', action='store_true')

    parser.add_argument('--random_label_init', action='store_true')

    parser.add_argument('--nyt_only_last_label_init', action='store_true')

    parser.add_argument('--only_test', action='store_true')
    parser.add_argument('--only_test_path', type=str, default=None)

    parser.add_argument('--rcv1_expand', type=str, default=None)
    parser.add_argument
    args = parser.parse_args()
    return args

def prepare(args):
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
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    logger.info("Training/evaluation parameters %s", args)

def test(args, best_macro_f1_path, best_micro_f1_path):
    bout = None
    for i, save_path in enumerate([best_micro_f1_path, best_macro_f1_path]):
        if save_path is None: continue
        flags = [
            '--tokenizer_name'         , args.model_name_or_path             ,
            '--input_file'             , args.test_file                  ,
            '--split'                  , 'test'                         ,
            '--do_lower_case'          ,
            '--model_path'             , str(save_path)              ,
            '--max_seq_length'         , str(args.max_source_seq_length + args.max_target_seq_length) if args.label_cpt_decodewithpos else str(args.max_source_seq_length)             ,
            '--max_tgt_length'         , str(args.max_target_seq_length)             ,
            '--batch_size'             , '128'                            ,
            '--beam_size'              , '1'                             ,
            '--length_penalty'         , '0'                             ,
            '--forbid_duplicate_ngrams',
            '--mode'                   , 's2s'                           ,
            '--forbid_ignore_word'     , '"."'                           ,
            '--cached_features_file'   , str(os.path.join(args.output_dir, "cached_features_for_test.pt")),
            '--add_vocab_file'         , args.add_vocab_file]

        if args.softmax_label_only:
            flags.append('--softmax_label_only')
        if args.soft_label:
            flags.append('--soft_label')
        if args.soft_label_hier_real:
            flags.append('--soft_label_hier_real_with_train_file')
            flags.append(args.train_file)
        if args.label_cpt_decodewithpos:
            flags.append('--target_no_offset')

        out = test_main(flags)
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
    args = get_args()
    prepare(args)
    if args.only_test:
        args.wandb = False
        test(args, args.only_test_path, None)
        exit(0)

    if args.wandb:
        wandb.init(
            project="HBGL",
            name=args.output_dir.split('/')[-1],
        )
        wandb.define_metric("train/global_step")
        wandb.define_metric("*", step_metric="train/global_step", step_sync=True)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
        # Make sure only the first process in distributed training will download model & vocab
    # Load pretrained model and tokenizer
    model, tokenizer, vs = get_model_and_tokenizer(args)

    if args.local_rank == 0:
        torch.distributed.barrier()
        # Make sure only the first process in distributed training will download model & vocab

    if args.cached_train_features_file is None:
        if not args.lmdb_cache:
            args.cached_train_features_file = os.path.join(args.output_dir, "cached_features_for_training.pt")
        else:
            args.cached_train_features_file = os.path.join(args.output_dir, "cached_features_for_training_lmdb")

    if args.soft_label:
        args.cached_train_features_file += 'soft_label'
        if args.soft_label_hier_real:
            hier_labels = None
            for line in open(args.train_file):
                if hier_labels:
                    for i, l in enumerate(json.loads(line)['tgt']):
                        hier_labels[i] |=  set(l)
                else:
                    hier_labels = [set(i) for i in json.loads(line)['tgt']]

            if hier_labels is None:
                raise ValueError("No hier_labels")

            hier_labels = [tokenizer.convert_tokens_to_ids(list([j.lower() for j in i])) for i in hier_labels]

            set_hier_labels_in_model(model, hier_labels)
            model.soft_label_hier_real = args.soft_label_hier_real
    
    training_features = load_and_cache_examples(
        example_file=args.train_file, tokenizer=tokenizer, local_rank=args.local_rank,
        cached_features_file=args.cached_train_features_file, shuffle=True,
        lmdb_cache=args.lmdb_cache, lmdb_dtype=args.lmdb_dtype,
        soft_label=args.soft_label,
    )

    if args.add_vocab_file:
        for i in training_features:
            for j in i.target_ids:
                if args.soft_label:
                    for ji in j:
                        assert ji >= vs

    best_macro_f1_path, best_micro_f1_path = train(args, training_features, model, tokenizer)
    if args.test_file:
        test(args, best_macro_f1_path, best_micro_f1_path)


if __name__ == "__main__":
    main()
