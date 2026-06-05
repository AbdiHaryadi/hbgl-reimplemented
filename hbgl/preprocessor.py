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

def remove_first_token(inputs: BatchEncoding[torch.Tensor]) -> BatchEncoding[torch.Tensor]:
    return BatchEncoding({k: v[:, 1:] for k, v in inputs.items()})

class HBGLPreprocessor:
    def __init__(
        self,
        bert_tokenizer: PreTrainedTokenizer,
        hierarchy: Hierarchy,
        device: torch.device
    ):
        labels = hierarchy.get_labels()
        bert_tokenizer.add_tokens(labels)
        label_ids = bert_tokenizer.convert_tokens_to_ids(labels)
        assert isinstance(label_ids, list)

        mask_token_id = bert_tokenizer.mask_token_id
        if not isinstance(mask_token_id, int):
            raise ValueError(f"Expecting integer ID of mask token, got {type(mask_token_id)}.")
        
        levels = hierarchy.get_levels()
        
        self._bert_tokenizer = bert_tokenizer
        self._mask_token_id = mask_token_id
        self._labels = labels
        self._hierarchy_label_ids = label_ids
        self._hierarchy_label_levels = levels
        self._device = device
    
    def get_num_tokenizer_tokens(self):
        return len(self._bert_tokenizer)
    
    def _assert_valid_label_paths(self, label_paths: list[list[str]]):
        for path in label_paths:
            for label in path:
                if label not in self._labels:
                    raise ValueError(f"Unknown label: {label}")
    
    def _prepare_input_ids(
        self,
        bert_text_input_ids: torch.Tensor,
        bert_label_input_ids: torch.Tensor,
    ):
        text_ids = bert_text_input_ids
        label_ids = bert_label_input_ids

        mask_token_id = self.get_mask_token_id()
        mask_ids = torch.full_like(label_ids, mask_token_id)

        return torch.cat([text_ids, label_ids, mask_ids], dim=1)
    
    def _prepare_label_token_type_ids(
        self,
        bert_label_token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        label_ids = torch.ones_like(bert_label_token_type_ids)
        return label_ids
    
    def _prepare_position_ids(self, bert_input_ids: torch.Tensor) -> torch.Tensor:
        batch_size = bert_input_ids.shape[0]
        device = self.get_device()
        assert device == bert_input_ids.device
        
        length = bert_input_ids.shape[1]
        ids = torch.arange(length, device=device)
        ids = ids.repeat(batch_size, 1)

        return ids
    
    def _prepare_label_position_ids(self, bert_label_input_ids: torch.Tensor) -> torch.Tensor:
        batch_size = bert_label_input_ids.shape[0]
        device = self.get_device()
        assert device == bert_label_input_ids.device

        length = bert_label_input_ids.shape[1]
        ids = torch.arange(length, device=device) + 1
        ids = ids.repeat(batch_size, 1)

        return ids
    
    def _prepare_labels(self, label_paths: list[list[str]]):
        device = self.get_device()
        return torch.tensor([
            [int(t in path) for t in self._labels]
            for path in label_paths
        ], device=device).float()
    
    def _pass_to_bert_tokenizer(self, texts: list[str]) -> BatchEncoding[torch.Tensor]:
        inputs = self._bert_tokenizer(texts, return_tensors="pt", padding=True)

        device = self.get_device()
        for k in inputs.keys():
            inputs[k] = inputs[k].to(device)

        return inputs
    
    def __call__(
        self,
        texts: list[str],
        label_paths: list[list[str]] | None = None,
    ) -> BatchEncoding[torch.Tensor]:
        bert_text_inputs = self._pass_to_bert_tokenizer(texts)
        position_ids = self._prepare_position_ids(
            bert_text_inputs["input_ids"],
        )
        text_data = {
            "input_ids": bert_text_inputs["input_ids"],
            "token_type_ids": bert_text_inputs["token_type_ids"],
            "attention_mask": bert_text_inputs["attention_mask"],
            "position_ids": position_ids,
        }
        if label_paths is None:
            label_data = {}
        else:
            self._assert_valid_label_paths(label_paths)
            bert_label_inputs = self._pass_to_bert_tokenizer([" ".join(p) for p in label_paths])
            bert_label_inputs = remove_first_token(bert_label_inputs)

            label_token_type_ids = self._prepare_label_token_type_ids(
                bert_label_inputs["token_type_ids"]
            )
            label_position_ids = self._prepare_label_position_ids(
                bert_label_inputs["input_ids"],
            )
            labels = self._prepare_labels(label_paths)
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
    
    def get_mask_token_id(self) -> int:
        return self._mask_token_id
    
    def get_device(self) -> torch.device:
        return self._device
