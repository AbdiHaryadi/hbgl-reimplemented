import logging

import torch
from transformers import BertConfig, BertTokenizer

from args import HBGLTrainingArguments
from config import BertForSeq2SeqConfig
from hierarchy import LabelMapper
import ghle
from modeling import BertForSequenceToSequence, BertForSequenceToSequenceWithPseudoMask

logger = logging.getLogger(__name__)

def get_model_and_tokenizer(args: HBGLTrainingArguments):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
        # Make sure only the first process in distributed training will download model & vocab
    # Load pretrained model and tokenizer
    model, tokenizer, old_vocab_size = _get_model_and_tokenizer(args)

    if args.local_rank == 0:
        torch.distributed.barrier()
        # Make sure only the first process in distributed training will download model & vocab

    return model,tokenizer,old_vocab_size

def _get_model_and_tokenizer(args: HBGLTrainingArguments):
    model_config = BertConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None)
    config = BertForSeq2SeqConfig.from_exist_config(
        config=model_config, label_smoothing=args.label_smoothing,
        fix_word_embedding=args.fix_word_embedding,
        max_position_embeddings=args.max_source_seq_length + args.max_target_seq_length)

    logger.info("Model config for seq2seq: %s", str(config))

    tokenizer = BertTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case, cache_dir=args.cache_dir if args.cache_dir else None)
    assert isinstance(tokenizer.mask_token_id, int)
    assert isinstance(tokenizer.sep_token_id, int)

    model: BertForSequenceToSequence = BertForSequenceToSequenceWithPseudoMask.from_pretrained(
        args.model_name_or_path, config=config,
        reuse_position_embedding=True,
        cache_dir=args.cache_dir if args.cache_dir else None)

    text_vocab_size = model.config.vocab_size
    if args.add_vocab_file:
        init_label_emb = init_label_name_from_averaging(args, tokenizer, model)

        label_mapper = LabelMapper.load(args.add_vocab_file)
        tokenizer.add_tokens([token.lower() for token in label_mapper.get_all_tokens()])

        # for compare with same seed
        rng_state = torch.get_rng_state()

        if args.label_cpt:
            init_label_emb = ghle.GHLETrainer(
                ghle.GHLETrainingArguments.from_args(args),
                tokenizer,
            ).train(init_label_emb)

        elif args.random_label_init:
            init_label_emb = torch.nn.Embedding(len(label_mapper), model.config.hidden_size).weight.data

        # for compare with same seed
        torch.set_rng_state(rng_state)

        model.concat_label_embeddings(init_label_emb)

    model.mask_token_id = tokenizer.mask_token_id
    model.sep_token_id = tokenizer.sep_token_id
    model.text_vocab_size = text_vocab_size

    return model, tokenizer, text_vocab_size

def init_label_name_from_averaging(
        args: HBGLTrainingArguments,
        tokenizer: BertTokenizer,
        model: BertForSequenceToSequence
):
    assert isinstance(tokenizer.pad_token_id, int)
    
    label_mapper = LabelMapper.load(args.add_vocab_file)
    label_name_tensors = []
    max_l = -1

    for label in label_mapper.get_all_labels():
            # When using BERT, lk should be lower()
        current_label_names = tokenizer.encode(label.lower(), add_special_tokens=False)
        label_name_tensors.append(current_label_names)
        max_l = max(len(current_label_names), max_l)
    label_name_tensors = torch.tensor([i + [tokenizer.pad_token_id] * (max_l - len(i)) for i in label_name_tensors], dtype=torch.long)

    with torch.no_grad():
        init_label_emb = model.bert.embeddings.word_embeddings(label_name_tensors)
        label_mask = label_name_tensors != tokenizer.pad_token_id
        init_label_emb = (label_mask.unsqueeze(-1) * init_label_emb).sum(1)
    return init_label_emb
