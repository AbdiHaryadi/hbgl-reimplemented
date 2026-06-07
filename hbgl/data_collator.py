import torch
from transformers import BatchEncoding

from hbgl.config import HBGLConfig
from hbgl.utils import (
    prepare_mask_inputs_from_label,
    prepare_token_type_ids_from_input_ids,
    prepare_position_ids_from_input_ids,
    prepare_4d_attention_mask,
)

class HBGLDataCollator:
    def __init__(self, config: HBGLConfig):
        self._mask_token_id = config.mask_token_id

    def __call__(self, features: list[BatchEncoding[list[int]]]) -> BatchEncoding[torch.Tensor]:
        # TODO: What if there is no labels?
        TEXT_TOKENIZATION_ATTRIBUTES = ["input_ids", "attention_mask"]
        LABEL_TOKENIZATION_ATTRIBUTES = ["label_input_ids", "label_attention_mask"]
        LABEL_ATTRIBUTE = "labels"
        ATTRIBUTES = TEXT_TOKENIZATION_ATTRIBUTES + LABEL_TOKENIZATION_ATTRIBUTES + [LABEL_ATTRIBUTE]
        inputs: dict[str, list[list[int]]] = {a: [] for a in ATTRIBUTES}

        max_text_length = 0
        max_label_length = 0
        for f in features:
            text_length = -1
            for a in TEXT_TOKENIZATION_ATTRIBUTES:
                inputs[a].append(f[a])
                length = len(f[a])
                if text_length == -1:
                    text_length = length
                else:
                    assert length == text_length
            max_text_length = max(text_length, max_text_length)

            label_length = -1
            for a in LABEL_TOKENIZATION_ATTRIBUTES:
                inputs[a].append(f[a])
                
                length = len(f[a])
                if label_length == -1:
                    label_length = length
                else:
                    assert length == label_length
            max_label_length = max(label_length, max_label_length)

            a = LABEL_ATTRIBUTE
            inputs[a].append(f[a])

        for a in TEXT_TOKENIZATION_ATTRIBUTES:
            for value in inputs[a]:
                while len(value) < max_text_length:
                    value.append(0)

        for a in LABEL_TOKENIZATION_ATTRIBUTES:
            for value in inputs[a]:
                while len(value) < max_label_length:
                    value.append(0)
        
        new_inputs: dict[str, torch.Tensor] = {}
        for a in inputs.keys():
            new_inputs[a] = torch.tensor(inputs[a])
        new_inputs[LABEL_ATTRIBUTE] = new_inputs[LABEL_ATTRIBUTE].float()

        text_input_ids = new_inputs["input_ids"]
        label_input_ids = new_inputs["label_input_ids"]

        label_attention_mask = new_inputs["label_attention_mask"]

        mask_input_ids, mask_attention_mask = prepare_mask_inputs_from_label(
            label_input_ids=label_input_ids,
            label_attention_mask=label_attention_mask,
            mask_token_id=self._mask_token_id
        )
        token_type_ids = prepare_token_type_ids_from_input_ids(
            text_input_ids=text_input_ids,
            label_input_ids=label_input_ids,
            mask_input_ids=mask_input_ids,
        )
        position_ids = prepare_position_ids_from_input_ids(
            text_input_ids=new_inputs["input_ids"],
            label_input_ids=new_inputs["label_input_ids"],
            mask_input_ids=mask_input_ids,
        )
        attention_mask = prepare_4d_attention_mask(
            text_attention_mask_2d=new_inputs["attention_mask"],
            label_attention_mask_2d=new_inputs["label_attention_mask"],
            mask_attention_mask_2d=mask_attention_mask,
        )
        labels = new_inputs[LABEL_ATTRIBUTE]

        return BatchEncoding({
            "text_input_ids": text_input_ids,
            "label_input_ids": label_input_ids,
            "mask_input_ids": mask_input_ids,
            "token_type_ids": token_type_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        })
