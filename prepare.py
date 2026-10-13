import logging
import pickle

import torch
from transformers import BertConfig, BertTokenizer, RobertaConfig

from args import HBGLTrainingArguments
from hierarchy import LabelHierarchy, LabelMapper
import ghle
from modeling import BertForSequenceToSequence, BertForSequenceToSequenceWithPseudoMask

logger = logging.getLogger(__name__)


class BertForSeq2SeqConfig(BertConfig):
    def __init__(self, label_smoothing=0.1, source_type_id=0, target_type_id=1, 
                 rel_pos_bins=0, max_rel_pos=0, fix_word_embedding=False, **kwargs):
        super(BertForSeq2SeqConfig, self).__init__(**kwargs)
        self.label_smoothing = label_smoothing
        self.source_type_id = source_type_id
        self.target_type_id = target_type_id
        self.max_rel_pos = max_rel_pos
        self.rel_pos_bins = rel_pos_bins
        self.fix_word_embedding = fix_word_embedding

    @classmethod
    def from_exist_config(cls, config, label_smoothing=0.1, max_position_embeddings=None, fix_word_embedding=False):
        required_keys = [
            "vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads",
            "hidden_act", "intermediate_size", "hidden_dropout_prob", "attention_probs_dropout_prob",
            "max_position_embeddings", "type_vocab_size", "initializer_range", "layer_norm_eps", 
            ]

        kwargs = {}
        for key in required_keys:
            assert hasattr(config, key)
            kwargs[key] = getattr(config, key)

        kwargs["vocab_size_or_config_json_file"] = kwargs["vocab_size"]
        if isinstance(config, RobertaConfig):
            kwargs["type_vocab_size"] = 0
            kwargs["max_position_embeddings"] = kwargs["max_position_embeddings"] - 2
        
        additional_keys = [
            "source_type_id", "target_type_id", "rel_pos_bins", "max_rel_pos", 
        ]
        for key in additional_keys:
            if hasattr(config, key):
                kwargs[key] = getattr(config, key)

        if max_position_embeddings is not None and max_position_embeddings > config.max_position_embeddings:
            kwargs["max_position_embeddings"] = max_position_embeddings
            logger.info("  **  Change max position embeddings to %d  ** " % max_position_embeddings)

        return cls(label_smoothing=label_smoothing, fix_word_embedding=fix_word_embedding, **kwargs)

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

    assert args.mask_way == "v2", "I don't implement BertForSequenceToSequenceUniLMV1 yet, so ... nope."
    logger.info("Construct model %s" % BertForSequenceToSequenceWithPseudoMask.MODEL_NAME)

    model: BertForSequenceToSequence = BertForSequenceToSequenceWithPseudoMask.from_pretrained(
        args.model_name_or_path, config=config,
        reuse_position_embedding=True,
        cache_dir=args.cache_dir if args.cache_dir else None)

    if args.add_vocab_file:
        label_mapper = LabelMapper.load(args.add_vocab_file)
        label_name_tensors = []
        max_l = -1
        if args.rcv1_expand:
            rcv1_label_expand = {}
            for parent in open(args.rcv1_expand):
                oi = [j for j in parent.replace('\n', '').split(' ') if len(j) > 0]
                rcv1_label_expand[oi[3]] = parent.split('child-description: ')[-1].lower().replace('\n', '')

        for label in label_mapper.get_all_labels():
            if args.one_by_one_label_init_map:
                one_by_one_label_init_map = (
                    LabelHierarchy.load(args.one_by_one_label_init_map)
                    .get_one_by_one_label_init_map()
                )
                print(f'map {label} to {one_by_one_label_init_map[label]}')
                label_name_tensors.append(tokenizer.encode(one_by_one_label_init_map[label], add_special_tokens=False))
            elif args.nyt_only_last_label_init:
                print(f'map {label} to {label.split("/")[-1]}')
                label_name_tensors.append(tokenizer.encode(label.split("/")[-1], add_special_tokens=False))
            elif args.rcv1_expand:
                print(f'map {label} to {rcv1_label_expand[label]}')
                label_name_tensors.append(tokenizer.encode(rcv1_label_expand[label], add_special_tokens=False))
            else:
                # When using BERT, lk should be lower()
                label_name_tensors.append(tokenizer.encode(label.lower(), add_special_tokens=False))
            max_l = max(len(label_name_tensors[-1]), max_l)
        label_name_tensors = torch.tensor([i + [tokenizer.pad_token_id] * (max_l - len(i)) for i in label_name_tensors], dtype=torch.long)

        with torch.no_grad():
            init_label_emb = model.bert.embeddings.word_embeddings(label_name_tensors)
            label_mask = label_name_tensors != tokenizer.pad_token_id
            init_label_emb = (label_mask.unsqueeze(-1) * init_label_emb).sum(1)
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
        old_vocab_size = model.config.vocab_size
        model.config.vocab_size = model.config.vocab_size + len(label_mapper)
    else:
        old_vocab_size = model.config.vocab_size

    model.soft_label = True
    model.mask_token_id = tokenizer.mask_token_id
    model.sep_token_id = tokenizer.sep_token_id
    model.vs = old_vocab_size

    return model, tokenizer, old_vocab_size
