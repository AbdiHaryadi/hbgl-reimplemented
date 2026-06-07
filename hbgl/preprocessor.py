from transformers import BatchEncoding, PreTrainedTokenizer

from hbgl.config import HBGLConfig

def remove_first_token(inputs: BatchEncoding) -> BatchEncoding:
    return BatchEncoding({k: v[1:] for k, v in inputs.items()})

class HBGLPreprocessor:
    def __init__(
        self,
        bert_tokenizer: PreTrainedTokenizer,
        config: HBGLConfig,
    ):
        label_list = config.label_list
        self._bert_tokenizer = bert_tokenizer
        self._label_list = label_list

        self._label_to_id_map = {label: i for i, label in enumerate(label_list)}

        self.config = config
    
    def _assert_valid_label_path(self, label_path: list[str]):
        for label in label_path:
            if label not in self._label_list:
                raise ValueError(f"Unknown label: {label}")
    
    def _prepare_labels(self, label_path: list[str]):
        return [int(t in label_path) for t in self._label_list]
    
    def _prepare_label_input_ids(self, label_path: list[str]):
        ids = [self._label_to_id_map[label] for label in label_path] 
        ids.append(self.config.sep_label_id)
        return ids
    
    def _prepare_label_attention_mask(self, input_ids: list[int]):
        return [1 for _ in input_ids]
    
    def _pass_to_bert_tokenizer(self, text: str) -> BatchEncoding:
        inputs = self._bert_tokenizer(text, truncation=True, max_length=self.config.max_length)
        return inputs
    
    def __call__(
        self,
        text: str,
        label_path: list[str] | None = None,
    ) -> BatchEncoding:
        bert_text_inputs = self._pass_to_bert_tokenizer(text)
        text_data = {
            "input_ids": bert_text_inputs["input_ids"],
            "attention_mask": bert_text_inputs["attention_mask"],
        }
        if label_path is None:
            label_data = {}
        else:
            labels = self._prepare_labels(label_path)
            label_input_ids = self._prepare_label_input_ids(label_path)
            label_attention_mask = self._prepare_label_attention_mask(label_input_ids)
            label_data = {
                "labels": labels,
                "label_input_ids": label_input_ids,
                "label_attention_mask": label_attention_mask,
            }

        return BatchEncoding(text_data | label_data)
