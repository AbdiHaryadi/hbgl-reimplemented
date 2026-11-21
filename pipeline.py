from dataclasses import dataclass
import logging
import os

import torch
from transformers import BertConfig, BertTokenizer

from evaluation import buf_to_traces, get_device, to_base_model
from hierarchy import LabelMapper
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

    def __call__(self, text: str):
        # Magic code probably to clear cache
        with torch.no_grad():
            torch.cuda.empty_cache()

        tokens = self.tokenize(text)
        batch_logits_per_level = self.get_batch_logits_per_level(tokens)

        return HBGLPipelineOutput(
            tokens=tokens,
            batch_logits_per_level=batch_logits_per_level,
            named_traces=self.get_named_traces(batch_logits_per_level),
            agrads=self.get_agrads(batch_logits_per_level),
        )

    def get_named_traces(self, current_batch_logits: list[torch.Tensor]):
        current_batch_predictions: list[torch.Tensor] = [
            (logits > 0).float().detach().cpu()
            for logits in current_batch_logits
        ]
        
        single_prediction_result = [predictions_per_level[0] for predictions_per_level in current_batch_predictions]
        named_traces = [[self.id2label[i - 1] if i > 0 else "[SEP]" for i in range(len(x[0])) if x[0, i] == 1] for x in single_prediction_result]
        return named_traces

    def tokenize(self, text: str):
        tokens = self.tokenizer.tokenize(text.strip().lower())[:self.max_src_length]
        tokens = [str(x) for x in tokens]
        return tokens

    def get_batch_logits_per_level(self, tokens: list[str]):
        buf = [tokens]
        # I don't care with Jiang result. I believe with the score.
        buf_to_traces(self.bi_uni_pipeline, self.model, buf)

        current_batch_logits = to_base_model(self.model).get_last_logit_list()
        assert current_batch_logits is not None
        return current_batch_logits

    def get_agrads(self, batch_logits_per_level: list[torch.Tensor]):
        current_batch_attention_probs_list = to_base_model(self.model).get_last_attention_probs_list()
        assert current_batch_attention_probs_list is not None
            
        for attention_probs in current_batch_attention_probs_list:
            attention_probs.retain_grad()
        
        loss_fct = torch.nn.BCEWithLogitsLoss()
        total_loss = torch.tensor(0.0, device=get_device())
        for logits in batch_logits_per_level:
            self_target = (logits > 0).float()
            total_loss += loss_fct(logits, self_target)

        total_loss.backward()

        current_batch_agrads: list[torch.Tensor] = []
        for attention_probs in current_batch_attention_probs_list:
            assert attention_probs.grad is not None
            ca = -(attention_probs[:,:,-1] * attention_probs.grad[:,:,-1]).mean(dim=1)
            current_batch_agrads.append(ca.detach().cpu())

        agrad_result = [agrads_per_level[0] for agrads_per_level in current_batch_agrads]
        return agrad_result

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
