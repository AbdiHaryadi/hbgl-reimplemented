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
        mask_ids = torch.full_like(label_ids, self._mask_token_id)

        return torch.cat([text_ids, label_ids, mask_ids], dim=1)
    
    def _prepare_token_type_ids(
        self,
        bert_text_token_type_ids: torch.Tensor,
        bert_label_token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        text_ids = bert_text_token_type_ids
        label_ids = torch.ones_like(bert_label_token_type_ids)
        mask_ids = label_ids

        return torch.cat([text_ids, label_ids, mask_ids], dim=1)
    
    def _prepare_attention_mask(
        self,
        bert_text_attention_mask: torch.Tensor,
        bert_label_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        bert_mask_attention_mask = bert_label_attention_mask
        bert_attention_mask = torch.cat(
            [
                bert_text_attention_mask,
                bert_label_attention_mask,
                bert_mask_attention_mask,
            ],
            dim=1
        )
        bert_attention_mask = bert_attention_mask.unsqueeze(1)
        bert_attention_mask = bert_attention_mask.unsqueeze(3)

        batch_size = bert_attention_mask.shape[0]
        device = self._device
        assert device == bert_attention_mask.device

        text_length = bert_text_attention_mask.shape[1]
        label_length = bert_label_attention_mask.shape[1]
        mask_length = label_length
        length = text_length + label_length + mask_length
        
        label_start_index = text_length
        label_stop_index = label_start_index + label_length
        mask_start_index = label_stop_index
        mask_stop_index = mask_start_index + mask_length

        label_base_subattention = torch.ones((label_length, label_length), device=device)
        label_to_label_subattention = torch.tril(label_base_subattention)
        label_to_mask_subattention = torch.tril(label_base_subattention, diagonal=-1)

        mask_to_mask_subattention = torch.eye(label_length, device=device)

        text_attention = torch.ones((batch_size, 1, length, text_length), device=device)

        label_attention = torch.zeros((batch_size, 1, length, label_length), device=device)
        label_attention[:, :, label_start_index:label_stop_index, :] = label_to_label_subattention
        label_attention[:, :, mask_start_index:mask_stop_index, :] = label_to_mask_subattention

        mask_attention = torch.zeros((batch_size, 1, length, mask_length), device=device)
        mask_attention[:, :, mask_start_index:mask_stop_index, :] = mask_to_mask_subattention

        result = torch.cat([text_attention, label_attention, mask_attention], dim=3)
        result = result * bert_attention_mask
        return result
    
    def _prepare_position_ids(
        self,
        bert_text_input_ids: torch.Tensor,
        bert_label_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = bert_text_input_ids.shape[0]
        assert batch_size == bert_label_input_ids.shape[0]
        
        device = self._device
        assert device == bert_text_input_ids.device
        assert device == bert_label_input_ids.device

        text_length = bert_text_input_ids.shape[1]
        text_ids = torch.arange(text_length, device=device)
        text_ids = text_ids.repeat(batch_size, 1)

        label_length = bert_label_input_ids.shape[1]
        label_ids = torch.arange(label_length, device=device) + 1
        label_ids = label_ids.repeat(batch_size, 1)

        mask_ids = label_ids

        return torch.cat([text_ids, label_ids, mask_ids], dim=1)
    
    def _prepare_labels(self, label_paths: list[list[str]]):
        return torch.tensor([
            [int(t in path) for t in self._labels]
            for path in label_paths
        ], device=self._device).float()
    
    def _pass_to_bert_tokenizer(self, texts: list[str]) -> BatchEncoding[torch.Tensor]:
        inputs = self._bert_tokenizer(texts, return_tensors="pt", padding=True)
        for k in inputs.keys():
            inputs[k] = inputs[k].to(self._device)

        return inputs
    
    def __call__(
        self,
        texts: list[str],
        label_paths: list[list[str]],
    ):
        # TODO: How do you handle if there is no label_paths?
        self._assert_valid_label_paths(label_paths)

        bert_text_inputs = self._pass_to_bert_tokenizer(texts)
        bert_label_inputs = self._pass_to_bert_tokenizer([" ".join(p) for p in label_paths])
        bert_label_inputs = remove_first_token(bert_label_inputs)

        input_ids = self._prepare_input_ids(
            bert_text_inputs["input_ids"],
            bert_label_inputs["input_ids"],
        )
        token_type_ids = self._prepare_token_type_ids(
            bert_text_inputs["token_type_ids"],
            bert_label_inputs["token_type_ids"],
        )
        attention_mask = self._prepare_attention_mask(
            bert_text_inputs["attention_mask"],
            bert_label_inputs["attention_mask"],
        )
        position_ids = self._prepare_position_ids(
            bert_text_inputs["input_ids"],
            bert_label_inputs["input_ids"],
        )
        labels = self._prepare_labels(label_paths)

        return BatchEncoding({
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "labels": labels,
        })
    
    def get_hierarchy_label_ids(self) -> list[int]:
        return self._hierarchy_label_ids.copy()
    
    def get_hierarchy_label_levels(self) -> list[int]:
        return self._hierarchy_label_levels.copy()
