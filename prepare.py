from collections import defaultdict
import logging
import pickle

import torch
from transformers import BertConfig, BertTokenizer, RobertaConfig

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

def get_model_and_tokenizer(args):
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
            label_map = pickle.load(f)
        label_tokens_start_index  = model.bert.embeddings.word_embeddings.num_embeddings
        labels_key = list(label_map.keys())
        label_name_tensors = []
        max_l = -1
        if args.rcv1_expand:
            rcv1_label_expand = {}
            for i in open(args.rcv1_expand):
                oi = [j for j in i.replace('\n', '').split(' ') if len(j) > 0]
                rcv1_label_expand[oi[3]] = i.split('child-description: ')[-1].lower().replace('\n', '')

        for lk in labels_key:
            if args.one_by_one_label_init_map:
                hiera = defaultdict(set)
                _label_dict = {}
                with open(args.one_by_one_label_init_map) as f:
                    _label_dict['Root'] = -1
                    for line in f.readlines():
                        line = line.strip().split('\t')
                        for i in line[1:]:
                            if i not in _label_dict:
                                _label_dict[i] = len(_label_dict) - 1
                            hiera[line[0]].add(i)
                    _label_dict.pop('Root')

                r_hiera = {}
                for i in hiera:
                    for j in list(hiera[i]):
                        r_hiera[j] = i

                def _loop(a):
                    if r_hiera[a] != 'Root':
                        return [a,] + _loop(r_hiera[a])
                    else:
                        return [a]

                one_by_one_label_init_map = {}
                for i in _label_dict:
                    one_by_one_label_init_map[i] = '/'.join(_loop(i)[::-1])
                print(f'map {lk} to {one_by_one_label_init_map[lk]}')
                label_name_tensors.append(tokenizer.encode(one_by_one_label_init_map[lk], add_special_tokens=False))
            elif args.nyt_only_last_label_init:
                print(f'map {lk} to {lk.split("/")[-1]}')
                label_name_tensors.append(tokenizer.encode(lk.split("/")[-1], add_special_tokens=False))
            elif args.rcv1_expand:
                print(f'map {lk} to {rcv1_label_expand[lk]}')
                label_name_tensors.append(tokenizer.encode(rcv1_label_expand[lk], add_special_tokens=False))
            else:
                label_name_tensors.append(tokenizer.encode(lk, add_special_tokens=False))
            max_l = max(len(label_name_tensors[-1]), max_l)
        label_name_tensors = torch.tensor([i + [tokenizer.pad_token_id] * (max_l - len(i)) for i in label_name_tensors], dtype=torch.long)

        with torch.no_grad():
            init_label_emb = model.bert.embeddings.word_embeddings(label_name_tensors)
            label_mask = label_name_tensors != tokenizer.pad_token_id
            init_label_emb = (label_mask.unsqueeze(-1) * init_label_emb).sum(1)
        label_tokens = [i for i in range(len(label_map))]
        tokenizer.add_tokens([label_map[label].lower() for label in labels_key])
        if args.label_cpt:
            # for compare with same seed
            rng_state = torch.get_rng_state()

            hiera, _label_dict = load_hierarchy_and_label_dict(args.label_cpt)
            r_hiera = {}
            for i in hiera:
                for j in list(hiera[i]):
                    r_hiera[j] = i

            def _loop(a):
                if r_hiera[a] != 'Root':
                    return [a,] + _loop(r_hiera[a])
                else:
                    return [a]

            label_class = {}
            for i in _label_dict:
                label_class[i] = len(_loop(i))
            # cls l1 l2 l3 sep
            attention_mask = torch.zeros((len(label_tokens) + 2, len(label_tokens) + 2))
            num_hiers = defaultdict(set)
            reversed_hiers = {}
            for hi in hiera:
                for hj in list(hiera[hi]):
                    def _label_map_f(x):
                        if x == 'Root': return -1
                        return int(label_map[x].replace('[A_', '').replace(']', ''))
                    attention_mask[_label_map_f(hi) + 1][_label_map_f(hj) + 1] = 1
                    num_hiers[_label_map_f(hi) + 1].add(_label_map_f(hj) + 1)
                    reversed_hiers[_label_map_f(hj) + 1] = _label_map_f(hi) + 1
                    if args.label_cpt_use_bce:
                        attention_mask[_label_map_f(hj) + 1][_label_map_f(hi) + 1] = 1
            input_ids = torch.tensor(tokenizer.encode(' '.join(label_map.values()).lower()), dtype=torch.long)
            assert len(input_ids) == len(labels_key) + 2
            position_ids = torch.tensor([0, ] + [label_class[i] for i in labels_key] + [max(label_class.values()) + 1,], dtype=torch.long)

            init_label_emb = ghle.train(args, tokenizer, input_ids, attention_mask,
                                            position_ids, init_label_emb, num_hiers, reversed_hiers).detach().cpu()

            # for compare with same seed
            torch.set_rng_state(rng_state)
        elif args.random_label_init:
            rng_state = torch.get_rng_state()
            init_label_emb = torch.nn.Embedding(len(label_tokens), model.config.hidden_size).weight.data
            torch.set_rng_state(rng_state)

        model.bert.embeddings.word_embeddings.weight.data = torch.cat([model.bert.embeddings.word_embeddings.weight.data, init_label_emb], dim=0)
        model.bert.embeddings.word_embeddings.num_embeddings += len(label_tokens)
        model.cls.predictions.decoder_weight.data = torch.cat([model.cls.predictions.decoder_weight.data, init_label_emb], dim=0)
        model.cls.predictions.bias.data =  torch.cat([model.cls.predictions.bias.data, torch.zeros(len(label_tokens))],
                                                        dim=0)
        vs = model.config.vocab_size
        model.config.vocab_size = model.config.vocab_size + len(label_tokens)
        if args.softmax_label_only:
            set_label_start_index_in_model(model, label_tokens_start_index)
    else:
        vs = model.config.vocab_size

    if args.soft_label:
        set_model_for_soft_label(model, tokenizer, vs)

    return model, tokenizer, vs

def load_hierarchy_and_label_dict(hierarchy_path):
    hiera = defaultdict(set)
    _label_dict = {}
    with open(hierarchy_path) as f:
        _label_dict['Root'] = -1
        for line in f.readlines():
            line = line.strip().split('\t')
            for token in line[1:]:
                if token not in _label_dict:
                    _label_dict[token] = len(_label_dict) - 1
                hiera[line[0]].add(token)
        _label_dict.pop('Root')
    return hiera,_label_dict

def set_label_start_index_in_model(model, label_tokens_start_index):
    model.label_start_index = label_tokens_start_index

def set_model_for_soft_label(model, tokenizer, vs):
    model.soft_label = True
    model.mask_token_id = tokenizer.mask_token_id
    model.sep_token_id = tokenizer.sep_token_id
    model.vs = vs