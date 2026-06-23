import torch
import torch.nn as nn
from transformers import PreTrainedModel

from hbgl.config import HBGLConfig
from hbgl.utils import (
    prepare_4d_boolean_attention_mask_with_size,
    prepare_position_ids_from_size,
    prepare_token_type_ids_from_size,
)

class HBGLModel(nn.Module):
    """
    Note: This model doesn't support multiple GPU.
    """

    def __init__(
        self,
        bert: PreTrainedModel,
        config: HBGLConfig,
        label_embeddings: torch.Tensor | None = None,
    ):
        super(HBGLModel, self).__init__()
        self.bert = bert

        device = self.bert.device
        bert_embeddings = self._get_word_embeddings()
        embedding_dim = bert_embeddings.embedding_dim
        if label_embeddings == None:
            label_embeddings = torch.randn(
                (config.label_count, embedding_dim),
                device=device
            )
        else:
            label_embeddings = label_embeddings.to(device)

        self.label_embeddings = nn.Parameter(label_embeddings)

        self.config = config
        self._device = device

    def _get_word_embeddings(self):
        embeddings = self.bert.get_input_embeddings()
        if not isinstance(embeddings, nn.Embedding):
            raise ValueError(f"bert.embeddings is not nn.Embedding (got: {type(embeddings)})")
        return embeddings

    def forward(
        self,
        text_input_ids: torch.Tensor,
        label_input_ids: torch.Tensor,
        mask_input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        position_ids: torch.Tensor,
    ):
        text_input_ids = text_input_ids.to(self._device)
        label_input_ids = label_input_ids.to(self._device)
        mask_input_ids = mask_input_ids.to(self._device)
        attention_mask = attention_mask.to(self._device)
        token_type_ids = token_type_ids.to(self._device)
        position_ids = position_ids.to(self._device)

        # Harusnya nanti di sini pakai input_embeds karena kita pakai label embeddings yang tidak ada dalam token.
        inputs_embeds = self._prepare_inputs_embeds(text_input_ids, label_input_ids, mask_input_ids)
        inputs_embeds = inputs_embeds.to(self._device)
        outputs = self.bert(
            inputs_embeds=inputs_embeds,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        last_hidden_state = outputs.last_hidden_state
        if not isinstance(last_hidden_state, torch.Tensor):
            raise ValueError("last_hidden_state is not a torch.Tensor")
        
        # Ini ukurannya [batch_size, text_len + label_len + mask_len, dimension]
        last_hidden_state = last_hidden_state.to(self._device)
        last_hidden_state = last_hidden_state[:, -mask_input_ids.shape[1]:]
        scores = torch.matmul(last_hidden_state, self.label_embeddings.T)
        scores = torch.sigmoid(scores)
        return scores
    
    def _forward_word_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        bert_embeddings: nn.Embedding = self._get_word_embeddings()
        return bert_embeddings(x)
    
    def _prepare_inputs_embeds(
        self,
        input_ids: torch.Tensor,
        label_input_ids: torch.Tensor,
        mask_input_ids: torch.Tensor,
    ):
        text_embeds = self._forward_word_embeddings(input_ids)

        label_input_ids = label_input_ids.clone()
        mask_in_label_input_ids = label_input_ids == self.config.sep_label_id
        label_input_ids[mask_in_label_input_ids] = 0
        label_embeds = self.label_embeddings[label_input_ids]
        label_embeds[mask_in_label_input_ids] = self._forward_word_embeddings(
            torch.tensor(self.config.sep_token_id, device=label_embeds.device)
        )

        mask_embeds: torch.Tensor = self._forward_word_embeddings(mask_input_ids)
        return torch.cat([
            text_embeds,
            label_embeds,
            mask_embeds,
        ], dim=1)

def generate(
        model: HBGLModel,
        config: HBGLConfig,
        input_ids: list[int],
        max_length: int = -1,
):
    device = model.bert.device

    text_input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    text_embeds = model._forward_word_embeddings(text_input_ids)
    label_embeds = torch.empty((1, 0, text_embeds.shape[-1]), device=device)
    mask_input_ids = torch.tensor([[config.mask_token_id]], dtype=torch.long, device=device)
    mask_embeds = model._forward_word_embeddings(mask_input_ids)

    stop = False
    scores = torch.empty((0, config.label_count), device=device)

    while not stop:
        inputs_embeds = torch.cat([text_embeds, label_embeds, mask_embeds], dim=1)

        bool_attention_mask = prepare_4d_boolean_attention_mask_with_size(
            batch_size=1,
            text_length=text_embeds.shape[1],
            label_length=label_embeds.shape[1],
            mask_length=mask_embeds.shape[1],
        )
        attention_mask = torch.where(
            bool_attention_mask.bool(),
            torch.tensor(0.0, device=bool_attention_mask.device),
            torch.tensor(-1e9, device=bool_attention_mask.device),
        )
        attention_mask = attention_mask.to(device)

        token_type_ids = prepare_token_type_ids_from_size(
            batch_size=1,
            text_length=text_embeds.shape[1],
            label_length=label_embeds.shape[1],
            mask_length=mask_embeds.shape[1],
        )
        token_type_ids = token_type_ids.to(device)

        position_ids = prepare_position_ids_from_size(
            batch_size=1,
            text_length=text_embeds.shape[1],
            label_length=label_embeds.shape[1],
            mask_length=mask_embeds.shape[1],
        )
        position_ids = position_ids.to(device)

        outputs = model.bert(
            inputs_embeds=inputs_embeds,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        last_hidden_state = outputs.last_hidden_state
        if not isinstance(last_hidden_state, torch.Tensor):
            raise ValueError("last_hidden_state is not a torch.Tensor")
        
        # Ini ukurannya [batch_size (1), text_len + label_len + mask_len (1), dimension]
        last_hidden_state = last_hidden_state[:, -1]
        # Now the dimension is only [1, dimension], want to be multiplied by [label_length, dimension]
        # Result should be [1, label_length]
        current_scores = torch.matmul(last_hidden_state, model.label_embeddings.T)
        current_scores = torch.sigmoid(current_scores)
        current_predictions = torch.where(current_scores > 0.5, True, False)
        current_sum_predictions = current_predictions.sum().item()
        
        scores = torch.cat([scores, current_scores], dim=0)

        if current_sum_predictions == 0 or scores.shape[0] == max_length:
            stop = True
        else:
            local_embeds = model.label_embeddings[current_predictions[0]].sum(dim=0)
            local_embeds = local_embeds.unsqueeze(0)
            local_embeds = local_embeds.unsqueeze(1)
            # Now the size should be [1, 1, dimension]

            # label_embeds: [1, ..., dimension]
            label_embeds = torch.cat([label_embeds, local_embeds], dim=1)

    return scores

def scores_to_labels(
        config: HBGLConfig,
        scores: torch.Tensor,
):
    result: list[int] = []
    for current_level_index, scores_at_current_level in enumerate(scores):
        current_level = current_level_index + 1
        for current_label_index, current_score in enumerate(scores_at_current_level):
            if current_score > 0.5 and current_level == config.label_levels[current_label_index]:
                result.append(current_label_index)
    
    return result
