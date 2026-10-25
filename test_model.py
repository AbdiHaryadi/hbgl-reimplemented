"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import json
import logging
import math
import os
import pickle

from tqdm import tqdm

import torch
from transformers import BertConfig, BertTokenizer

from args import HBGLTestingArguments, parse_to_testing_arguments
from eval import evaluate
from hierarchy import LabelMapper
from modeling_decoding import BertForSeq2SeqDecoder
import seq2seq_loader
from utils import load_and_cache_examples, load_and_cache_examples_fast


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def detokenize(tk_list):
    r_list = []
    for tk in tk_list:
        if tk.startswith('##') and len(r_list) > 0:
            r_list[-1] = r_list[-1] + tk[2:]
        else:
            r_list.append(tk)
    return r_list

def test(args: HBGLTestingArguments):
    if args.max_tgt_length >= args.max_seq_length - 2:
        raise ValueError("Maximum tgt length exceeds max seq length - 2.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer: BertTokenizer = BertTokenizer.from_pretrained(
        args.tokenizer_name, do_lower_case=True,
        cache_dir=args.cache_dir if args.cache_dir else None)
    assert isinstance(tokenizer.mask_token_id, int)
    assert isinstance(tokenizer.sep_token_id, int)
    assert isinstance(tokenizer.cls_token, str)
    assert isinstance(tokenizer.sep_token, str)
    assert isinstance(tokenizer.pad_token, str)

    label_mapper = LabelMapper.load(args.add_vocab_file)
    tokenizer.add_tokens([token.lower() for token in label_mapper.get_all_tokens()])

    tokenizer.model_max_length = args.max_seq_length
    mask_word_id = tokenizer.mask_token_id
    eos_word_id = tokenizer.sep_token_id
    print(args.model_path)
    found_checkpoint_flag = False
    for model_recover_path in glob.glob(args.model_path):
        if not os.path.isdir(model_recover_path):
            continue

        logger.info("***** Recover model: %s *****", model_recover_path)

        config_file = args.config_path if args.config_path else os.path.join(model_recover_path, "config.json")
        logger.info("Read decoding config from: %s" % config_file)
        config = BertConfig.from_json_file(config_file)

        def indexer(tokens: list[str]) -> list[int]:
            result = tokenizer.convert_tokens_to_ids(tokens)
            assert isinstance(result, list)
            return result

        bi_uni_pipeline = [
            seq2seq_loader.Preprocess4Seq2seqDecoder(
                tokenizer_vocab_words(tokenizer), indexer, args.max_seq_length,
                max_tgt_length=args.max_tgt_length,
                source_type_id=config.source_type_id, target_type_id=config.target_type_id,
                cls_token=tokenizer.cls_token, sep_token=tokenizer.sep_token, pad_token=tokenizer.pad_token
            )
        ]

        found_checkpoint_flag = True
        model = BertForSeq2SeqDecoder.from_pretrained(
            model_recover_path, config=config,
            mask_word_id=mask_word_id,
            eos_id=eos_word_id,
        )

        model.label_start_index = model.bert.embeddings.word_embeddings.num_embeddings - len(label_mapper)

        model.to(device)  # type: ignore
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

        output_lines = get_outputs(args, device, tokenizer, bi_uni_pipeline, model)
        if args.output_file:
            fn_out = args.output_file
        else:
            fn_out = model_recover_path+'.'+args.split
        with open(fn_out, "w", encoding="utf-8") as fout:
            for l in output_lines:
                fout.write(l)
                fout.write("\n")
        
        def token_to_id(token):
            token = token.lower()
            try:
                token = int(token.replace('[a_', '').replace(']', ''))
                token = 0 if token >= len(label_to_id_map) else token
                return token
            except:
                return 0

        predict_labels = [
            (label_str.split(' ') if label_str else []) for label_str in
            [i.replace("\n", '') for i in output_lines]
        ]
        predict_labels = [list(set([token_to_id(j) for j  in i])) for i in predict_labels]
        with open(args.input_file) as f:
            ground_labels = [list(set(sum(json.loads(i)["tgt"], []))) for i in f]

        with open(args.add_vocab_file, 'rb') as f:
            label_to_id_map = pickle.load(f)
        id2label = {v: k for k, v in label_to_id_map.items()}
        out = evaluate(
            epoch_predicts=predict_labels,
            epoch_labels=ground_labels,
            id2label=id2label,
        )
        del out['full']
        print(out)
        return out

    if not found_checkpoint_flag:
        logger.info("Not found the model checkpoint file!")

def get_outputs(
        args: HBGLTestingArguments,
        device: torch.device,
        tokenizer: BertTokenizer,
        bi_uni_pipeline: list[seq2seq_loader.Preprocess4Seq2seqDecoder],
        model: BertForSeq2SeqDecoder | torch.nn.DataParallel[BertForSeq2SeqDecoder],
):
    torch.cuda.empty_cache()
    model.eval()
    max_src_length = args.max_seq_length - 2 - args.max_tgt_length

    num_lines = sum(1 for _ in open(args.input_file))
    if num_lines < 10000:
        to_pred = load_and_cache_examples(
                args.input_file, tokenizer, local_rank=-1,
                cached_features_file=args.cached_features_file, shuffle=False)
        # Kinda inconsistent because eval_mode is utilized in fast mode.
    else:
        to_pred = load_and_cache_examples_fast(
                args.input_file, tokenizer, local_rank=-1,
                cached_features_file=args.cached_features_file, shuffle=False, eval_mode=True)

    input_lines: list[list[str]] = []
    for line in to_pred:
        tokens = tokenizer.convert_ids_to_tokens(line.source_ids)[:max_src_length]
        assert isinstance(tokens, list)
        input_lines.append(tokens)
    if args.subset > 0:
        logger.info("Decoding subset: %d", args.subset)
        input_lines = input_lines[:args.subset]

    input_lines_with_indices = sorted(
        list(enumerate(input_lines)),
        key=lambda x: -len(x[1])
    )
    output_lines = [""] * len(input_lines_with_indices)
    total_batch = math.ceil(len(input_lines_with_indices) / args.batch_size)

    with tqdm(total=total_batch) as pbar:
        batch_count = 0
        first_batch = True
        next_i = 0
        while next_i < len(input_lines_with_indices):
            _chunk = input_lines_with_indices[next_i:next_i + args.batch_size]
            buf_id = [x[0] for x in _chunk]
            buf = [x[1] for x in _chunk]
            next_i += args.batch_size
            batch_count += 1
            max_a_len = max([len(x) for x in buf])
            instances: list[seq2seq_loader.Preprocess4Seq2seqDecoderOutputType] = []
            for instance in [(x, max_a_len) for x in buf]:
                for proc in bi_uni_pipeline:
                    instances.append(proc(instance))
            with torch.no_grad():
                input_ids, token_type_ids, position_ids, input_mask, _, task_idx = seq2seq_loader.batch_list_to_batch_tensors_for_testing(instances)
                traces = model(
                    input_ids=input_ids.to(device),
                    token_type_ids=token_type_ids.to(device),
                    position_ids=position_ids.to(device),
                    attention_mask=input_mask.to(device),
                    task_idx=task_idx.to(device)
                )

                output_ids = traces.tolist()
                for i in range(len(buf)):
                    w_ids = output_ids[i]
                    output_buf = tokenizer.convert_ids_to_tokens(w_ids)
                    output_tokens = []
                    for t in output_buf:
                        if t in (tokenizer.sep_token, tokenizer.pad_token):
                            break
                        output_tokens.append(t)
                    output_sequence = ' '.join(detokenize(output_tokens))
                    if '\n' in output_sequence:
                        output_sequence = " [X_SEP] ".join(output_sequence.split('\n'))
                    output_lines[buf_id[i]] = output_sequence
                    if first_batch or batch_count % 50 == 0:
                        logger.info("{} = {}".format(buf_id[i], output_sequence))
            pbar.update(1)
            first_batch = False
    return output_lines

def tokenizer_vocab_words(tokenizer: BertTokenizer):
    max_id = max(tokenizer.vocab.values())
    vocab_words = [f"[unused{id}]" for id in range(max_id + 1)]
    for token, id in tokenizer.vocab.items():
        vocab_words[id] = token
    return vocab_words

if __name__ == "__main__":
    print(test(parse_to_testing_arguments()))
