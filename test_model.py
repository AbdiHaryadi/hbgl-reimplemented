"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import json
import logging
import math
import os

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


def detokenize(tk_list: list[str]):
    r_list: list[str] = []
    for tk in tk_list:
        if tk.startswith('##') and len(r_list) > 0:
            r_list[-1] = r_list[-1] + tk[2:]
        else:
            r_list.append(tk)
    return r_list

def test(args: HBGLTestingArguments):
    print(args.model_path)
    found_checkpoint_flag = False
    for model_recover_path in glob.glob(args.model_path):
        if not os.path.isdir(model_recover_path):
            continue

        found_checkpoint_flag = True
        logger.info("***** Recover model: %s *****", model_recover_path)

        out = test_inner(args, model_recover_path)
        del out['full']
        print(out)
        return out

    if not found_checkpoint_flag:
        logger.info("Not found the model checkpoint file!")

def test_inner(args: HBGLTestingArguments, model_recover_path: str):
    tokenizer, bi_uni_pipeline, model = prepare_for_get_outputs(args, model_recover_path)
    output_lines = get_outputs(args, tokenizer, bi_uni_pipeline, model)
    if args.output_file:
        fn_out = args.output_file
    else:
        fn_out = model_recover_path+'.'+args.split
    with open(fn_out, "w", encoding="utf-8") as fout:
        for l in output_lines:
            fout.write(l)
            fout.write("\n")

    label_mapper = LabelMapper.load(args.add_vocab_file)
    def token_to_id(token):
        token = token.lower()
        try:
            token = int(token.replace('[a_', '').replace(']', ''))
            token = 0 if token >= len(label_mapper) else token
            return token
        except:
            return 0

    predict_labels = [
        (label_str.split(' ') if label_str else []) for label_str in
        [i.replace("\n", '') for i in output_lines]
    ]
    predict_labels = [list(set([token_to_id(j) for j  in i])) for i in predict_labels]
    ground_labels = get_ground_flattened_labels(args.input_file)

    id2label = {label_mapper.label_to_id(label): label for label in label_mapper.get_all_labels()}
    out = evaluate(
        epoch_predicts=predict_labels,
        epoch_labels=ground_labels,
        id2label=id2label,
    )
    return out

def get_ground_flattened_labels(input_file: str) -> list[list[int]]:
    with open(input_file) as f:
        ground_labels = [list(set(sum(json.loads(i)["tgt"], []))) for i in f]
    return ground_labels

def test_inner_for_get_scores(args: HBGLTestingArguments, model_path: str):
    tokenizer, bi_uni_pipeline, model = prepare_for_get_outputs(args, model_path)
    scores = get_scores(args, tokenizer, bi_uni_pipeline, model)
    ground_labels = get_ground_flattened_labels(args.input_file)
    outputs: list[list[float] | None] = []
    for current_scores, current_ground_labels in zip(scores, ground_labels):
        if current_scores is None:
            outputs.append(None)
            continue

        # current_scores selalu dimensinya [1, label_vocab_size + 1], dengan indeks 0 adalah EOS.
        outputs.append([
            max([score_per_level[0, label_id + 1].item() for score_per_level in current_scores])
            for label_id in current_ground_labels
        ])

    return outputs, ground_labels

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prepare_for_get_outputs(args: HBGLTestingArguments, model_recover_path: str):
    if args.max_tgt_length >= args.max_seq_length - 2:
        raise ValueError("Maximum tgt length exceeds max seq length - 2.")

    device = get_device()
    
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

    model = BertForSeq2SeqDecoder.from_pretrained(
        model_recover_path, config=config,
        mask_word_id=mask_word_id,
        eos_id=eos_word_id,
    )

    model.label_start_index = model.bert.embeddings.word_embeddings.num_embeddings - len(label_mapper)

    model.to(device)  # type: ignore
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    return tokenizer,bi_uni_pipeline,model

def get_outputs(
        args: HBGLTestingArguments,
        tokenizer: BertTokenizer,
        bi_uni_pipeline: list[seq2seq_loader.Preprocess4Seq2seqDecoder],
        model: BertForSeq2SeqDecoder | torch.nn.DataParallel[BertForSeq2SeqDecoder],
):
    torch.cuda.empty_cache()
    model.eval()
    input_lines_with_indices = get_input_lines_with_indices(args, tokenizer)
    output_lines = [""] * len(input_lines_with_indices)

    batch_count = 0
    first_batch = True
    for _chunk in iter_chunk(input_lines_with_indices, batch_size=args.batch_size):
        buf_id = [x[0] for x in _chunk]
        buf = [x[1] for x in _chunk]
        batch_count += 1
        with torch.no_grad():
            traces = buf_to_traces(bi_uni_pipeline, model, buf)

        output_ids = traces.tolist()
        for i in range(len(buf)):
            w_ids = output_ids[i]
            output_sequence = convert_w_ids_to_output_sequence(tokenizer, w_ids)
            output_lines[buf_id[i]] = output_sequence
            if first_batch or batch_count % 50 == 0:
                logger.info("{} = {}".format(buf_id[i], output_sequence))
        first_batch = False
    return output_lines

def get_scores(
        args: HBGLTestingArguments,
        tokenizer: BertTokenizer,
        bi_uni_pipeline: list[seq2seq_loader.Preprocess4Seq2seqDecoder],
        model: BertForSeq2SeqDecoder | torch.nn.DataParallel[BertForSeq2SeqDecoder],
):
    torch.cuda.empty_cache()
    model.eval()
    input_lines_with_indices = get_input_lines_with_indices(args, tokenizer)
    result: list[list[torch.Tensor] | None] = [None for _ in input_lines_with_indices]

    for _chunk in iter_chunk(input_lines_with_indices, batch_size=args.batch_size):
        buf_id = [x[0] for x in _chunk]
        buf = [x[1] for x in _chunk]
        with torch.no_grad():
            # I don't care about the output.
            buf_to_traces(bi_uni_pipeline, model, buf)

        current_batch_scores = to_base_model(model).get_last_score_list()
        if current_batch_scores is not None:
            for i in range(len(buf)):
                result[buf_id[i]] = [scores_per_level[i] for scores_per_level in current_batch_scores]

    return result

def to_base_model(model: BertForSeq2SeqDecoder | torch.nn.DataParallel[BertForSeq2SeqDecoder]):
    if isinstance(model, torch.nn.DataParallel):
        return model.module
    return model

def get_input_lines_with_indices(args: HBGLTestingArguments, tokenizer: BertTokenizer):
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
    return input_lines_with_indices

def iter_chunk(input_lines_with_indices: list[tuple[int, list[str]]], batch_size: int):
    total_batch = math.ceil(len(input_lines_with_indices) / batch_size)
    with tqdm(total=total_batch) as pbar:
        next_i = 0
        while next_i < len(input_lines_with_indices):
            _chunk = input_lines_with_indices[next_i:next_i + batch_size]
            yield _chunk
            next_i += batch_size
            pbar.update(1)

def buf_to_traces(
        bi_uni_pipeline: list[seq2seq_loader.Preprocess4Seq2seqDecoder],
        model: BertForSeq2SeqDecoder | torch.nn.DataParallel[BertForSeq2SeqDecoder],
        buf: list[list[str]],
) -> torch.Tensor:
    device = get_device()
    max_a_len = max([len(x) for x in buf])
    instances: list[seq2seq_loader.Preprocess4Seq2seqDecoderOutputType] = []
    for instance in [(x, max_a_len) for x in buf]:
        for proc in bi_uni_pipeline:
            instances.append(proc(instance))

    input_ids, token_type_ids, position_ids, input_mask, _, task_idx = seq2seq_loader.batch_list_to_batch_tensors_for_testing(instances)
    return model(
        input_ids=input_ids.to(device),
        token_type_ids=token_type_ids.to(device),
        position_ids=position_ids.to(device),
        attention_mask=input_mask.to(device),
        task_idx=task_idx.to(device)
    )

def convert_w_ids_to_output_sequence(tokenizer: BertTokenizer, w_ids: list[int]):
    output_buf = tokenizer.convert_ids_to_tokens(w_ids)
    assert isinstance(output_buf, list)
    output_tokens: list[str] = []
    for t in output_buf:
        if t in (tokenizer.sep_token, tokenizer.pad_token):
            break
        output_tokens.append(t)
    output_sequence = ' '.join(detokenize(output_tokens))
    if '\n' in output_sequence:
        output_sequence = " [X_SEP] ".join(output_sequence.split('\n'))
    return output_sequence

def tokenizer_vocab_words(tokenizer: BertTokenizer):
    max_id = max(tokenizer.vocab.values())
    vocab_words = [f"[unused{id}]" for id in range(max_id + 1)]
    for token, id in tokenizer.vocab.items():
        vocab_words[id] = token
    return vocab_words

if __name__ == "__main__":
    print(test(parse_to_testing_arguments()))
