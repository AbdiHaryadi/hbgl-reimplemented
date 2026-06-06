import torch
from transformers import BatchEncoding, PreTrainedTokenizer

class Hierarchy:
    def __init__(self, label_level_map: dict[str, int]) -> None:
        self._labels = list(label_level_map)
        self._levels = [label_level_map[label] for label in self._labels]
    
    def get_labels(self):
        return self._labels.copy()
    
    def get_levels(self):
        return self._levels.copy()

def remove_first_token(inputs: BatchEncoding) -> BatchEncoding:
    return BatchEncoding({k: v[1:] for k, v in inputs.items()})

class HBGLPreprocessor:
    def __init__(
        self,
        bert_tokenizer: PreTrainedTokenizer,
        hierarchy: Hierarchy,
    ):
        labels = hierarchy.get_labels()
        bert_tokenizer.add_tokens(labels)
        label_ids = bert_tokenizer.convert_tokens_to_ids(labels)
        assert isinstance(label_ids, list)
        
        levels = hierarchy.get_levels()
        
        self._bert_tokenizer = bert_tokenizer
        self._labels = labels
        self._hierarchy_label_ids = label_ids
        self._hierarchy_label_levels = levels
    
    def get_num_tokenizer_tokens(self):
        return len(self._bert_tokenizer)
    
    def _assert_valid_label_path(self, label_path: list[str]):
        for label in label_path:
            if label not in self._labels:
                raise ValueError(f"Unknown label: {label}")
    
    def _prepare_label_token_type_ids(
        self,
        bert_label_token_type_ids: list[int],
    ) -> list[int]:
        return [1 for _ in range(len(bert_label_token_type_ids))]
    
    def _prepare_position_ids(self, bert_input_ids: list[int]) -> list[int]:
        return [i for i in range(len(bert_input_ids))]
    
    def _prepare_label_position_ids(self, bert_label_input_ids: list[int]) -> list[int]:
        return [i + 1 for i in range(len(bert_label_input_ids))]
    
    def _prepare_labels(self, label_path: list[str]):
        return [int(t in label_path) for t in self._labels]
    
    def _pass_to_bert_tokenizer(self, text: str) -> BatchEncoding:
        inputs = self._bert_tokenizer(text, truncation=True)
        return inputs
    
    def __call__(
        self,
        text: str,
        label_path: list[str] | None = None,
    ) -> BatchEncoding:
        bert_text_inputs = self._pass_to_bert_tokenizer(text)
        position_ids = self._prepare_position_ids(
            bert_text_inputs["input_ids"],
        )
        text_data = {
            "input_ids": bert_text_inputs["input_ids"],
            "token_type_ids": bert_text_inputs["token_type_ids"],
            "attention_mask": bert_text_inputs["attention_mask"],
            "position_ids": position_ids,
        }
        if label_path is None:
            label_data = {}
        else:
            self._assert_valid_label_path(label_path)
            bert_label_inputs = self._pass_to_bert_tokenizer(" ".join(label_path))
            bert_label_inputs = remove_first_token(bert_label_inputs)

            label_token_type_ids = self._prepare_label_token_type_ids(
                bert_label_inputs["token_type_ids"]
            )
            label_position_ids = self._prepare_label_position_ids(
                bert_label_inputs["input_ids"],
            )
            labels = self._prepare_labels(label_path)
            label_data = {
                "labels": labels,
                "label_input_ids": bert_label_inputs["input_ids"],
                "label_token_type_ids": label_token_type_ids,
                "label_attention_mask": bert_label_inputs["attention_mask"],
                "label_position_ids": label_position_ids
            }

        return BatchEncoding(text_data | label_data)
    
    def get_hierarchy_label_ids(self) -> list[int]:
        return self._hierarchy_label_ids.copy()
    
    def get_hierarchy_label_levels(self) -> list[int]:
        return self._hierarchy_label_levels.copy()
