import logging
import pickle

import torch
from transformers import BertConfig, BertTokenizer, RobertaConfig

from args import HBGLTrainingArguments
from hierarchy import LabelHierarchy
from modeling import BertForSequenceToSequenceUniLMV1, BertForSequenceToSequenceWithPseudoMask
import ghle

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

    model_class = \
        BertForSequenceToSequenceWithPseudoMask if args.mask_way == 'v2' \
            else BertForSequenceToSequenceUniLMV1

    logger.info("Construct model %s" % model_class.MODEL_NAME)

    model = model_class.from_pretrained(
        args.model_name_or_path, config=config,
        reuse_position_embedding=True,
        cache_dir=args.cache_dir if args.cache_dir else None)

    if args.add_vocab_file:
        with open(args.add_vocab_file, 'rb') as f:
            label_to_id_map: dict[str, int] = pickle.load(f)
        label_tokens_start_index  = model.bert.embeddings.word_embeddings.num_embeddings
        labels_key = list(label_to_id_map.keys())
        label_name_tensors = []
        max_l = -1
        if args.rcv1_expand:
            rcv1_label_expand = {}
            for parent in open(args.rcv1_expand):
                oi = [j for j in parent.replace('\n', '').split(' ') if len(j) > 0]
                rcv1_label_expand[oi[3]] = parent.split('child-description: ')[-1].lower().replace('\n', '')

        for lk in labels_key:
            if args.one_by_one_label_init_map:
                hiera = LabelHierarchy.load(args.one_by_one_label_init_map, label_to_id_map)
                _label_index_map = hiera.get_label_index_map()

                one_by_one_label_init_map = {}
                for label in _label_index_map:
                    one_by_one_label_init_map[label] = '/'.join(hiera.find_path(label))
                print(f'map {lk} to {one_by_one_label_init_map[lk]}')
                label_name_tensors.append(tokenizer.encode(one_by_one_label_init_map[lk], add_special_tokens=False))
            elif args.nyt_only_last_label_init:
                print(f'map {lk} to {lk.split("/")[-1]}')
                label_name_tensors.append(tokenizer.encode(lk.split("/")[-1], add_special_tokens=False))
            elif args.rcv1_expand:
                print(f'map {lk} to {rcv1_label_expand[lk]}')
                label_name_tensors.append(tokenizer.encode(rcv1_label_expand[lk], add_special_tokens=False))
            else:
                # When using BERT, lk should be lower()
                label_name_tensors.append(tokenizer.encode(lk.lower(), add_special_tokens=False))
            max_l = max(len(label_name_tensors[-1]), max_l)
        label_name_tensors = torch.tensor([i + [tokenizer.pad_token_id] * (max_l - len(i)) for i in label_name_tensors], dtype=torch.long)

        with torch.no_grad():
            init_label_emb = model.bert.embeddings.word_embeddings(label_name_tensors)
            label_mask = label_name_tensors != tokenizer.pad_token_id
            init_label_emb = (label_mask.unsqueeze(-1) * init_label_emb).sum(1)
        tokenizer.add_tokens([id_to_label_token(label_to_id_map[label]).lower() for label in labels_key])

        # for compare with same seed
        rng_state = torch.get_rng_state()

        if args.label_cpt:
            hiera = LabelHierarchy.load(args.label_cpt, label_to_id_map)
            _label_index_map = hiera.get_label_index_map()

            label_depth = {token: len(hiera.find_path(token)) for token in _label_index_map}

            # cls l1 l2 l3 sep
            attention_mask = torch.zeros((len(label_to_id_map) + 2, len(label_to_id_map) + 2))
            for parent_id, child_id in hiera.iter_hiera_parent_child_id_pairs():
                # Here we use +1 because CLS is in index zero, and the first ID in hierarchy label is also zero.
                attention_mask[parent_id + 1][child_id + 1] = 1
                if args.label_cpt_use_bce:
                    attention_mask[child_id + 1][parent_id + 1] = 1
            
            input_ids = torch.tensor(tokenizer.encode(' '.join([id_to_label_token(id_) for id_ in label_to_id_map.values()]).lower()), dtype=torch.long)
            assert len(input_ids) == len(labels_key) + 2
            position_ids = torch.tensor([0, ] + [label_depth[i] for i in labels_key] + [max(label_depth.values()) + 1,], dtype=torch.long)

            # Selanjutnya validasi ini. :)
            init_label_emb = ghle.ghle_train(args, tokenizer, input_ids, attention_mask,
                                            position_ids, init_label_emb, hiera).detach().cpu()

        elif args.random_label_init:
            init_label_emb = torch.nn.Embedding(len(label_to_id_map), model.config.hidden_size).weight.data

        # for compare with same seed
        torch.set_rng_state(rng_state)

        model.bert.embeddings.word_embeddings.weight.data = torch.cat([model.bert.embeddings.word_embeddings.weight.data, init_label_emb], dim=0)
        model.bert.embeddings.word_embeddings.num_embeddings += len(label_to_id_map)
        model.cls.predictions.decoder_weight.data = torch.cat([model.cls.predictions.decoder_weight.data, init_label_emb], dim=0)
        model.cls.predictions.bias.data =  torch.cat([model.cls.predictions.bias.data, torch.zeros(len(label_to_id_map))],
                                                        dim=0)
        vs = model.config.vocab_size
        model.config.vocab_size = model.config.vocab_size + len(label_to_id_map)
        if args.softmax_label_only:
            set_label_start_index_in_model(model, label_tokens_start_index)
    else:
        vs = model.config.vocab_size

    if args.soft_label:
        set_model_for_soft_label(model, tokenizer, vs)

    return model, tokenizer, vs

def set_label_start_index_in_model(model, label_tokens_start_index):
    model.label_start_index = label_tokens_start_index

def set_model_for_soft_label(model, tokenizer, vs):
    model.soft_label = True
    model.mask_token_id = tokenizer.mask_token_id
    model.sep_token_id = tokenizer.sep_token_id
    model.vs = vs

def id_to_label_token(id_: int):
    return f"[A_{id_}]"
