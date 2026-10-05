import argparse
from dataclasses import dataclass

import torch

@dataclass
class HBGLTrainingArguments:
    # Required parameters
    train_file: str
    valid_file: str
    model_name_or_path: str
    output_dir: str

    # Other parameters
    test_file: str | None = None
    log_dir: str | None = None

    config_name: str | None = None
    tokenizer_name: str | None = None
    cache_dir: str | None = None
    max_source_seq_length: int = 464
    max_target_seq_length: int = 48
    cached_train_features_file: str | None = None
    do_lower_case: bool = False
    per_gpu_train_batch_size: int = 8
    learning_rate: float = 5e-05
    gradient_accumulation_steps: int = 1
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-08
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.1
    num_training_steps: int = -1
    num_training_epochs: int = 10
    num_warmup_steps: int = 0
    random_prob: float = 0.1
    keep_prob: float = 0.1
    fix_word_embedding: bool = False
    logging_steps: int = 500
    save_steps: int = 1500
    no_cuda: bool = False
    seed: int = 42
    local_rank: int = -1
    source_mask_prob: float = -1.0
    target_mask_prob: float = 0.5
    num_max_mask_token: int = 0
    mask_way: str = 'v2'
    lmdb_cache: bool = False
    lmdb_dtype: str = 'h'
    add_vocab_file: str | None = None
    wandb: bool = False
    softmax_label_only: bool = False
    soft_label: bool = False
    soft_label_hier_real: bool = False
    one_by_one_label_init_map: str | None = None
    label_cpt: str | None = None
    label_cpt_lr: float = 0.001
    label_cpt_steps: int = 500
    label_cpt_bsz: int = 32
    label_cpt_not_incr_mask_ratio: bool = False
    label_cpt_use_bce: bool = False
    label_cpt_decodewithpos: bool = False
    random_label_init: bool = False
    nyt_only_last_label_init: bool = False
    only_test: bool = False
    only_test_path: str | None = None
    rcv1_expand: str | None = None

    @property
    def n_gpu(self):
        if self.local_rank == -1 or self.no_cuda:
            return torch.cuda.device_count()
        else:
            return 1

    @property
    def device(self):
        if self.local_rank == -1 or self.no_cuda:
            return torch.device("cuda" if torch.cuda.is_available() and not self.no_cuda else "cpu")
        else:
            return torch.device("cuda", self.local_rank)

def parse_to_training_arguments(args=None):
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

    return HBGLTrainingArguments(**{
        k: v for k, v in vars(parser.parse_args(args=args)).items()
        if v is not None
    })
