from dataclasses import dataclass
import logging
import os

import torch
from transformers import BertConfig, BertTokenizer

from evaluation import buf_to_traces, get_device, to_base_model
from hierarchy import LabelHierarchy, LabelMapper
from modeling_decoding import BertForSeq2SeqDecoder
import seq2seq_loader
from utils import tokenizer_vocab_words

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class HBGLPipelineArguments:
    # Required parameters
    tokenizer_name: str
    add_vocab_file: str
    hierarchy_path: str

    # Other parameters
    config_path: str | None = None
    max_seq_length: int = 512

    max_tgt_length: int = 128

@dataclass
class HBGLPipelineOutput:
    tokens: list[str]
    batch_logits_per_level: list[torch.Tensor]
    named_traces: list[list[str]]
    agrads: list[torch.Tensor]
    traces: list[list[int]]
    agrads_loss: float

    @property
    def logits(self):
        return [
            logits.float().detach().cpu()
            for logits in self.batch_logits_per_level
        ]

    @property
    def predictions(self):
        result: list[str] = []
        for trace in self.named_traces:
            if (len(trace) == 0) or (len(trace) == 1 and trace[0] == "[SEP]"):
                break

            for label in trace:
                if label not in result and label != "[SEP]":
                    result.append(label)
        return result

    @property
    def scores(self):
        return [
            torch.sigmoid(logits).float().detach().cpu()
            for logits in self.batch_logits_per_level
        ]

    @property
    def prediction_ids(self):
        result: list[int] = []
        for trace in self.traces:
            if (len(trace) == 0) or (len(trace) == 1 and trace[0] == -1):
                break

            for label_id in trace:
                if label_id not in result and label_id != -1:
                    result.append(label_id)
        return result

class HBGLPipeline:
    def __init__(self, args: HBGLPipelineArguments, model_path: str):
        max_src_length = args.max_seq_length - 2 - args.max_tgt_length
        tokenizer, bi_uni_pipeline, model = prepare_for_get_outputs(args, model_path)
        model.eval()

        self.tokenizer = tokenizer
        self.max_src_length = max_src_length
        self.bi_uni_pipeline = bi_uni_pipeline
        self.model = model

        label_mapper = LabelMapper.load(args.add_vocab_file)
        self.id2label = {label_mapper.label_to_id(label): label
                            for label in label_mapper.get_all_labels()}

        self.max_tgt_length = args.max_tgt_length
        self.hierarchy = LabelHierarchy.load(args.hierarchy_path)

    def __call__(self, texts: str | list[str], compute_agrad: bool = True):
        with torch.no_grad():
            torch.cuda.empty_cache()

        if isinstance(texts, str):
            texts = [texts]
        
        tokens_list = [self.tokenize(text) for text in texts]

        if compute_agrad:
            batch_logits_per_level = self.get_batch_logits_per_level(tokens_list)
            agrads_list, agrads_loss_list = self.get_agrads_list(batch_logits_per_level)
        else:
            with torch.no_grad():
                batch_logits_per_level = self.get_batch_logits_per_level(tokens_list)
            agrads_list = [[] for _ in texts]
            agrads_loss_list = [-1.0 for _ in texts]

        named_traces_list = self.get_named_traces_list(batch_logits_per_level)
        traces_list = self.get_traces_list(batch_logits_per_level)

        result: list[HBGLPipelineOutput] = []
        for batch_index in range(len(texts)):
            result.append(HBGLPipelineOutput(
                tokens=tokens_list[batch_index],
                batch_logits_per_level=batch_logits_per_level[batch_index:batch_index+1],
                named_traces=named_traces_list[batch_index],
                agrads=agrads_list[batch_index],
                traces=traces_list[batch_index],
                agrads_loss=agrads_loss_list[batch_index],
            ))

        return result

    def tokenize(self, text: str):
        tokens = self.tokenizer.tokenize(text.strip().lower())[:self.max_src_length]
        tokens = [str(x) for x in tokens]
        return tokens

    def get_batch_logits_per_level(self, tokens_list: list[list[str]]):
        buf = tokens_list
        # I don't care with Jiang result. I believe with the score.
        buf_to_traces(self.bi_uni_pipeline, self.model, buf)

        current_batch_logits = to_base_model(self.model).get_last_logit_list()
        assert current_batch_logits is not None
        return current_batch_logits

    def get_agrads_list(self, batch_logits_per_level: list[torch.Tensor]):
        batch_size = len(batch_logits_per_level[0])
        assert batch_size > 0

        current_batch_attention_probs_list = to_base_model(self.model).get_last_attention_probs_list()
        assert current_batch_attention_probs_list is not None
            
        for attention_probs in current_batch_attention_probs_list:
            attention_probs.retain_grad()
        
        loss_fct = torch.nn.BCEWithLogitsLoss(reduction="none")
        total_loss = torch.zeros(batch_size, device=get_device())
        batch_mask = torch.ones(batch_size, 1, 1).to(get_device())
        for level_index, logits in enumerate(batch_logits_per_level):
            # Size of logits: <batch_size, 1, 1 + label_count>
            self_target = (logits > 0).float()

            mask = torch.zeros_like(logits[:, :, 1:], dtype=torch.float)
            for label_id, label in self.id2label.items():
                if self.hierarchy.get_label_depth(label) == level_index + 1:
                    mask[:, :, label_id] = 1
            mask = mask.to(get_device())

            # I think we don't care about sep token in the first element, so don't use it as a loss.
            # Ukuran logits sekarang: <batch_size, seq_length (1), label_vocab + 1>
            raw_loss: torch.Tensor = loss_fct(logits[:, :, 1:], self_target[:, :, 1:])
            total_loss += (raw_loss * mask * batch_mask).sum((1, 2))

            batch_mask = batch_mask * torch.amax(
                self_target[:, :, 1:] * mask,
                dim=2,
                keepdim=True,
            )
            if batch_mask.sum() == 0:
                break

        total_loss.sum().backward()

        current_batch_agrads: list[torch.Tensor] = []
        for attention_probs in current_batch_attention_probs_list:
            if attention_probs.grad is None:
                attention_probs_grad = torch.zeros_like(attention_probs[:,:,-1])
            else:
                attention_probs_grad = attention_probs.grad[:,:,-1]
            ca = -(attention_probs[:,:,-1] * attention_probs_grad).mean(dim=1)
            current_batch_agrads.append(ca.detach().cpu())

        total_loss = total_loss.detach().cpu()
        agrads_list: list[list[torch.Tensor]] = []
        loss_list: list[float] = []
        for batch_index, single_total_loss in enumerate(total_loss):
            single_agrad_result = [
                agrads_per_level[batch_index]
                for agrads_per_level in current_batch_agrads
            ]
            agrads_list.append(single_agrad_result)
            loss_list.append(float(single_total_loss.item()))

        return agrads_list, loss_list

    def get_named_traces_list(self, current_batch_logits: list[torch.Tensor]):
        current_batch_predictions: list[torch.Tensor] = [
            (logits > 0).float().detach().cpu()
            for logits in current_batch_logits
        ]

        named_traces_list: list[list[list[str]]] = []
        for batch_index in range(len(current_batch_predictions[0])):
            single_prediction_result = [predictions_per_level[batch_index] for predictions_per_level in current_batch_predictions]
            named_traces = [
                [self.id2label[i - 1] if i > 0 else "[SEP]"
                 for i in range(len(x[0])) if x[0, i] == 1]
                for x in single_prediction_result
            ]
            named_traces_list.append(named_traces)
        return named_traces_list

    def get_traces_list(self, current_batch_logits: list[torch.Tensor]):
        current_batch_predictions: list[torch.Tensor] = [
            (logits > 0).float().detach().cpu()
            for logits in current_batch_logits
        ]

        traces_list: list[list[list[int]]] = []
        for batch_index in range(len(current_batch_predictions[0])):
            single_prediction_result = [predictions_per_level[batch_index] for predictions_per_level in current_batch_predictions]
            traces = [[i - 1 for i in range(len(x[0])) if x[0, i] == 1] for x in single_prediction_result]
            traces_list.append(traces)
        
        return traces_list

    def get_loss(self, output: HBGLPipelineOutput, target_ids: list[list[int]]):
        loss_fct = torch.nn.BCEWithLogitsLoss(reduction="none")
        total_loss = torch.tensor(0.0, device=get_device())
        last_iteration = False
        for level_index, logits in enumerate(output.batch_logits_per_level):
            target = torch.zeros_like(logits[:, :, 1:], dtype=torch.float)
            if level_index == len(target_ids) or len(target_ids[level_index]) == 0:
                last_iteration = True
            else:
                for target_id in target_ids[level_index]:
                    target[:, :, target_id] = 1
            target = target.to(get_device())

            mask = torch.zeros_like(logits[:, :, 1:], dtype=torch.float)
            for label_id, label in self.id2label.items():
                if self.hierarchy.get_label_depth(label) == level_index + 1:
                    mask[:, :, label_id] = 1
            mask = mask.to(get_device())

            raw_loss: torch.Tensor = loss_fct(logits[:, :, 1:], target)
            total_loss += (raw_loss * mask).sum()

            if last_iteration:
                break

        return float(total_loss.detach().cpu().item())

def prepare_for_get_outputs(args: HBGLPipelineArguments, model_recover_path: str):
    if args.max_tgt_length >= args.max_seq_length - 2:
        raise ValueError("Maximum tgt length exceeds max seq length - 2.")

    device = get_device()
    
    tokenizer: BertTokenizer = BertTokenizer.from_pretrained(
        args.tokenizer_name, do_lower_case=True)
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
