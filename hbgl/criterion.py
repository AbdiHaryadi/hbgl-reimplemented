import torch
from torch.nn import BCEWithLogitsLoss
from transformers import PreTrainedModel

from hbgl.preprocessor import HBGLPreprocessor

class HBGLLossCalculator:
    def __init__(
        self,
        bert_masked_lm_model: PreTrainedModel,
        preprocessor: HBGLPreprocessor,
    ):
        num_tokenizer_tokens = preprocessor.get_num_tokenizer_tokens()
        bert_masked_lm_model.resize_token_embeddings(num_tokenizer_tokens)

        all_label_levels = preprocessor.get_hierarchy_label_levels()
        max_hierarchy_level = max(all_label_levels)
        label_level_indices = [
            -max_hierarchy_level + (level - 1) - 1
            # Perhatikan -1 terakhir. Itu berarti SEP terakhir tidak digunakan.

            for level in all_label_levels
        ]

        all_label_ids = preprocessor.get_hierarchy_label_ids()
        bce = BCEWithLogitsLoss(reduction="sum")
        device = bert_masked_lm_model.device
        assert device == preprocessor.get_device()

        self._bert_model = bert_masked_lm_model
        self._level_indices = label_level_indices
        self._label_token_indices = all_label_ids
        self._bce = bce
        self._mask_token_id = preprocessor.get_mask_token_id()
        self._device = device

    def __call__(
            self,
            input_ids: torch.Tensor,
            token_type_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            position_ids: torch.Tensor,
            labels: torch.Tensor,
            label_input_ids: torch.Tensor,
            label_token_type_ids: torch.Tensor,
            label_attention_mask: torch.Tensor,
            label_position_ids: torch.Tensor,
    ):
        # input_ids
        mask_token_id = self._mask_token_id
        mask_input_ids = torch.full_like(label_input_ids, mask_token_id)
        input_ids = torch.cat([input_ids, label_input_ids, mask_input_ids], dim=1)

        # token_type_ids
        mask_token_type_ids = label_token_type_ids
        token_type_ids = torch.cat([token_type_ids, label_token_type_ids, mask_token_type_ids], dim=1)

        # attention_mask
        text_attention_mask = attention_mask
        mask_attention_mask = label_attention_mask
        attention_mask = torch.cat([text_attention_mask, label_attention_mask, mask_attention_mask], dim=1)
        attention_mask = attention_mask.unsqueeze(1)
        attention_mask = attention_mask.unsqueeze(3)

        batch_size = attention_mask.shape[0]
        device = self._device
        assert device == attention_mask.device

        text_length = text_attention_mask.shape[1]
        label_length = label_attention_mask.shape[1]
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

        new_attention_mask = torch.cat([text_attention, label_attention, mask_attention], dim=3)
        new_attention_mask = new_attention_mask * attention_mask
        attention_mask = new_attention_mask

        # position_ids
        mask_position_ids = label_position_ids
        position_ids = torch.cat([position_ids, label_position_ids, mask_position_ids], dim=1)

        outputs = self._bert_model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids
        )
        preds = outputs.logits[:, self._level_indices, self._label_token_indices]
        return self._bce(preds, labels)
