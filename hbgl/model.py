import torch
from transformers import PreTrainedModel

from hbgl.preprocessor import HBGLPreprocessor

class HBGLModel:
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

        self._bert_model = bert_masked_lm_model
        self._level_indices = label_level_indices
        self._label_token_indices = all_label_ids

    def __call__(
            self,
            input_ids: torch.Tensor,
            token_type_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            position_ids: torch.Tensor,
    ):
        outputs = self._bert_model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids
        )
        preds = outputs.logits[:, self._level_indices, self._label_token_indices]
        return preds
