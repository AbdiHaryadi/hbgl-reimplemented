import torch
import torch.nn as nn
from transformers import PreTrainedModel

from hbgl.config import HBGLConfig

class HBGLModel(nn.Module):
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
