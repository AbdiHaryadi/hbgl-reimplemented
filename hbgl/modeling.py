import torch
import torch.nn as nn
from transformers import PreTrainedModel

from hbgl.config import HBGLConfig

def batch_arange_like(x: torch.Tensor):
    batch_size = x.shape[0]
    length = x.shape[1]

    x = torch.arange(length, device=x.device)
    x = x.repeat(batch_size, 1)
    return x

class HBGLModel(nn.Module):
    def __init__(
        self,
        bert: PreTrainedModel,
        config: HBGLConfig,
    ):
        super(HBGLModel, self).__init__()
        self.bert = bert

        bert_embeddings: nn.Embedding = self._get_word_embeddings(bert)
        self.word_embeddings = bert_embeddings

        embedding_dim = bert_embeddings.embedding_dim
        self.label_embeddings = nn.Parameter(
            torch.randn((config.label_count, embedding_dim))
        )

        self.config = config

    def _get_word_embeddings(self, model: PreTrainedModel):
        embeddings = model.get_input_embeddings()
        if not isinstance(embeddings, nn.Embedding):
            raise ValueError(f"bert.embeddings is not nn.Embedding (got: {type(embeddings)})")
        return embeddings

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            label_input_ids: torch.Tensor,
            label_attention_mask: torch.Tensor,
    ):
        mask_input_ids, mask_attention_mask = self._prepare_mask_inputs(label_input_ids, label_attention_mask)

        # Harusnya nanti di sini pakai input_embeds karena kita pakai label embeddings yang tidak ada dalam token.
        inputs_embeds = self._prepare_inputs_embeds(input_ids, label_input_ids, mask_input_ids)
        token_type_ids = self._prepare_token_type_ids(input_ids, label_input_ids, mask_input_ids)
        position_ids = self._prepare_position_ids(input_ids, label_input_ids, mask_input_ids)

        attention_mask = self._prepare_attention_mask(attention_mask, label_attention_mask, mask_attention_mask)        

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
        last_hidden_state = last_hidden_state[:, -mask_attention_mask.shape[1]:]
        scores = torch.matmul(last_hidden_state, self.label_embeddings.T)
        scores = torch.sigmoid(scores)
        return scores
    
    def _prepare_mask_inputs(
        self,
        label_input_ids: torch.Tensor,
        label_attention_mask: torch.Tensor,
    ):
        input_ids = torch.full_like(label_input_ids, self.config.mask_token_id)
        attention_mask = label_attention_mask
        return input_ids, attention_mask
    
    def _prepare_inputs_embeds(
        self,
        input_ids: torch.Tensor,
        label_input_ids: torch.Tensor,
        mask_input_ids: torch.Tensor,
    ):
        text_embeds: torch.Tensor = self.word_embeddings(input_ids)

        mask_in_label_input_ids = label_input_ids == self.config.sep_label_id
        label_input_ids[mask_in_label_input_ids] = 0
        label_embeds = self.label_embeddings[label_input_ids]
        label_embeds[mask_in_label_input_ids] = self.word_embeddings(
            torch.tensor(self.config.sep_token_id, device=label_embeds.device)
        )

        mask_embeds: torch.Tensor = self.word_embeddings(mask_input_ids)
        return torch.cat([
            text_embeds,
            label_embeds,
            mask_embeds,
        ], dim=1)
    
    def _prepare_token_type_ids(
        self,
        input_ids: torch.Tensor,
        label_input_ids: torch.Tensor,
        mask_input_ids: torch.Tensor,
    ):
        text_type_ids = torch.zeros_like(input_ids)
        label_type_ids = torch.ones_like(label_input_ids)
        mask_type_ids = torch.ones_like(mask_input_ids)
        return torch.cat([text_type_ids, label_type_ids, mask_type_ids], dim=1)
    
    def _prepare_position_ids(
        self,
        input_ids: torch.Tensor,
        label_input_ids: torch.Tensor,
        mask_input_ids: torch.Tensor,
    ):
        text_position_ids = batch_arange_like(input_ids)
        label_position_ids = batch_arange_like(label_input_ids) + 1
        mask_position_ids = batch_arange_like(mask_input_ids) + 1

        return torch.cat([text_position_ids, label_position_ids, mask_position_ids], dim=1)
    
    def _prepare_attention_mask(
        self,
        text_attention_mask_2d: torch.Tensor,
        label_attention_mask_2d: torch.Tensor,
        mask_attention_mask_2d: torch.Tensor,
    ):
        base_attention_mask = torch.cat([text_attention_mask_2d, label_attention_mask_2d, mask_attention_mask_2d], dim=1)
        base_attention_mask = base_attention_mask.unsqueeze(1)

        first_attention_mask = base_attention_mask.unsqueeze(3)
        second_attention_mask = base_attention_mask.unsqueeze(2)

        batch_size = base_attention_mask.shape[0]

        text_length = text_attention_mask_2d.shape[1]
        label_length = label_attention_mask_2d.shape[1]
        mask_length = mask_attention_mask_2d.shape[1]
        length = text_length + label_length + mask_length
        
        label_start_index = text_length
        label_stop_index = label_start_index + label_length
        mask_start_index = label_stop_index
        mask_stop_index = mask_start_index + mask_length

        label_base_subattention = torch.ones((label_length, label_length))
        label_to_label_subattention = torch.tril(label_base_subattention)

        label_to_mask_subattention = torch.tril(label_base_subattention, diagonal=-1)
        label_to_mask_subattention = label_to_mask_subattention[:mask_length]

        mask_to_mask_subattention = torch.eye(mask_length)

        text_attention = torch.ones((batch_size, 1, length, text_length))

        label_attention = torch.zeros((batch_size, 1, length, label_length))
        label_attention[:, :, label_start_index:label_stop_index, :] = label_to_label_subattention
        label_attention[:, :, mask_start_index:mask_stop_index, :] = label_to_mask_subattention

        mask_attention = torch.zeros((batch_size, 1, length, mask_length))
        mask_attention[:, :, mask_start_index:mask_stop_index, :] = mask_to_mask_subattention

        new_attention_mask = torch.cat([text_attention, label_attention, mask_attention], dim=3)
        new_attention_mask = new_attention_mask * first_attention_mask * second_attention_mask
        text_attention_mask_2d = new_attention_mask.bool()
        return text_attention_mask_2d
