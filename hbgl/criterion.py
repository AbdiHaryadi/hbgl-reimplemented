import torch
from torch.nn import BCELoss

from hbgl.config import HBGLConfig

class HBGLLoss:
    def __init__(
        self,
        config: HBGLConfig,
    ):
        self._label_token_indices = [i for i in range(config.label_count)]
        self._label_level_indices = self._prepare_label_level_indices(config)
        self._bce = BCELoss(reduction="sum")
    
    def _prepare_label_level_indices(
        self,
        config: HBGLConfig
    ):
        levels = config.label_levels
        label_level_indices = [
            level - 1
            # Perhatikan bahwa SEP terakhir tidak digunakan.
            for level in levels
        ]
        return label_level_indices
    
    def __call__(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        num_items_in_batch = None,  # This is unused tbh
    ):
        preds = outputs[:, self._label_level_indices, self._label_token_indices]
        return self._bce(preds, labels)
