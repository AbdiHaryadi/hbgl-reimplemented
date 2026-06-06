import torch
from transformers import PreTrainedTokenizer

class HBGLDataCollator:
    def __init__(self, tokenizer: PreTrainedTokenizer):
        mask_token_id = tokenizer.mask_token_id
        if not isinstance(mask_token_id, int):
            raise ValueError(f"Expecting integer ID of mask token, got {type(mask_token_id)}.")
        
        pad_token_id = tokenizer.pad_token_id
        if not isinstance(pad_token_id, int):
            raise ValueError(f"Expecting integer ID of pad token, got {type(pad_token_id)}.")

        pad_token_type_id = tokenizer.pad_token_type_id

        self._mask_token_id = mask_token_id
        self._pad_token_id = pad_token_id
        self._pad_token_type_id = pad_token_type_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        # TODO: What if there is no labels?
        TEXT_TOKENIZATION_ATTRIBUTES = ["input_ids", "token_type_ids", "attention_mask", "position_ids"]
        LABEL_TOKENIZATION_ATTRIBUTES = ["label_input_ids", "label_token_type_ids", "label_attention_mask", "label_position_ids"]
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
                    if a == "input_ids":
                        value.append(self._pad_token_id)
                    elif a == "token_type_ids":
                        value.append(self._pad_token_type_id)
                    elif a == "position_ids":
                        value.append(value[-1] + 1)
                    else:
                        value.append(0)

        for a in LABEL_TOKENIZATION_ATTRIBUTES:
            for value in inputs[a]:
                while len(value) < max_label_length:
                    value.append(0)

        # input_ids
        mask_token_id = self._mask_token_id
        text_input_ids = torch.tensor(inputs["input_ids"])
        label_input_ids = torch.tensor(inputs["label_input_ids"])
        mask_input_ids = torch.full_like(label_input_ids, mask_token_id)
        input_ids = torch.cat([text_input_ids, label_input_ids, mask_input_ids], dim=1)

        # token_type_ids
        text_token_type_ids = torch.tensor(inputs["token_type_ids"])
        label_token_type_ids = torch.tensor(inputs["label_token_type_ids"])
        mask_token_type_ids = label_token_type_ids
        token_type_ids = torch.cat([text_token_type_ids, label_token_type_ids, mask_token_type_ids], dim=1)

        # attention_mask
        text_attention_mask = torch.tensor(inputs["attention_mask"])
        label_attention_mask = torch.tensor(inputs["label_attention_mask"])
        mask_attention_mask = label_attention_mask
        attention_mask = torch.cat([text_attention_mask, label_attention_mask, mask_attention_mask], dim=1)
        attention_mask = attention_mask.unsqueeze(1)

        first_attention_mask = attention_mask.unsqueeze(3)
        second_attention_mask = attention_mask.unsqueeze(2)

        batch_size = attention_mask.shape[0]

        text_length = text_attention_mask.shape[1]
        label_length = label_attention_mask.shape[1]
        mask_length = label_length
        length = text_length + label_length + mask_length
        
        label_start_index = text_length
        label_stop_index = label_start_index + label_length
        mask_start_index = label_stop_index
        mask_stop_index = mask_start_index + mask_length

        label_base_subattention = torch.ones((label_length, label_length))
        label_to_label_subattention = torch.tril(label_base_subattention)
        label_to_mask_subattention = torch.tril(label_base_subattention, diagonal=-1)

        mask_to_mask_subattention = torch.eye(label_length)

        text_attention = torch.ones((batch_size, 1, length, text_length))

        label_attention = torch.zeros((batch_size, 1, length, label_length))
        label_attention[:, :, label_start_index:label_stop_index, :] = label_to_label_subattention
        label_attention[:, :, mask_start_index:mask_stop_index, :] = label_to_mask_subattention

        mask_attention = torch.zeros((batch_size, 1, length, mask_length))
        mask_attention[:, :, mask_start_index:mask_stop_index, :] = mask_to_mask_subattention

        new_attention_mask = torch.cat([text_attention, label_attention, mask_attention], dim=3)
        new_attention_mask = new_attention_mask * first_attention_mask * second_attention_mask
        attention_mask = new_attention_mask.bool()

        # position_ids
        text_position_ids = torch.tensor(inputs["position_ids"])
        label_position_ids = torch.tensor(inputs["label_position_ids"])
        mask_position_ids = label_position_ids
        position_ids = torch.cat([text_position_ids, label_position_ids, mask_position_ids], dim=1)

        # labels
        labels = torch.tensor(inputs["labels"]).float()

        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "labels": labels,
        }
