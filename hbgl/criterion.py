import torch
from torch.nn import BCEWithLogitsLoss
from transformers import PreTrainedTokenizer
from transformers.modeling_outputs import MaskedLMOutput

from hbgl.preprocessor import Hierarchy

class HBGLLoss:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        hierarchy: Hierarchy,
    ):
        self._label_token_indices = self._prepare_label_token_indices(tokenizer,  hierarchy)
        self._label_level_indices = self._prepare_label_level_indices(hierarchy)
        self._bce = BCEWithLogitsLoss(reduction="sum")

    def _prepare_label_token_indices(
        self,
        tokenizer: PreTrainedTokenizer,
        hierarchy: Hierarchy,
    ) -> list[int]:
        labels = hierarchy.get_labels()
        label_ids = tokenizer.convert_tokens_to_ids(labels)
        assert isinstance(label_ids, list)
        assert len(label_ids) == len(labels)
        return label_ids
    
    def _prepare_label_level_indices(
        self,
        hierarchy: Hierarchy,
    ):
        levels = hierarchy.get_levels()
        max_hierarchy_level = max(levels)
        label_level_indices = [
            -max_hierarchy_level + (level - 1) - 1
            # Perhatikan -1 terakhir. Itu berarti SEP terakhir tidak digunakan.

            for level in levels
        ]
        return label_level_indices
    
    def __call__(
        self,
        outputs: MaskedLMOutput,
        labels: torch.Tensor,
        num_items_in_batch = None,  # This is unused tbh
    ):
        if outputs.logits is None:
            raise ValueError("No logits found in outputs")
        
        preds = outputs.logits[:, self._label_level_indices, self._label_token_indices]
        return self._bce(preds, labels)
