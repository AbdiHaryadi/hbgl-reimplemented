# coding=utf-8
"""PyTorch BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import logging
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertIntermediate, BertOnlyMLMHead, BertOutput, BertPooler, BertPreTrainedModel, BertSelfOutput

from modeling import BertLayerNorm, relative_position_bucket

logger = logging.getLogger(__name__)

def log_variable(var_name: str, var_value):
    if isinstance(var_value, torch.Tensor):
        print(f"{var_name} ({type(var_value)}): {var_value.size()}")
    else:
        print(f"{var_name} ({type(var_value)}): {var_value}")

class HBGLBertSelfAttentionForDecoding(nn.Module):
    def __init__(self, config):
        super(HBGLBertSelfAttentionForDecoding, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, history_states=None,
                key_history=None, value_history=None,
                key_cache=None, value_cache=None, rel_pos=None):
        mixed_query_layer = self.query(hidden_states)

        if history_states is None:
            # possible issue: https://github.com/NVIDIA/apex/issues/131
            mixed_key_layer = F.linear(hidden_states, self.key.weight)
            mixed_value_layer = self.value(hidden_states)
        else:
            x_states = torch.cat((history_states, hidden_states), dim=1)
            # possible issue: https://github.com/NVIDIA/apex/issues/131
            mixed_key_layer = F.linear(x_states, self.key.weight)
            mixed_value_layer = self.value(x_states)

        if key_cache is not None and isinstance(key_cache, list):
            key_cache.append(mixed_key_layer)
            mixed_key_layer = torch.cat(key_cache, dim=1)

        if value_cache is not None and isinstance(value_cache, list):
            value_cache.append(mixed_value_layer)
            mixed_value_layer = torch.cat(value_cache, dim=1)

        return self.multi_head_attention(
            mixed_query_layer, mixed_key_layer, mixed_value_layer,
            attention_mask, rel_pos=rel_pos,
            key_history=key_history, value_history=value_history)

    def multi_head_attention(self, query, key, value, attention_mask, rel_pos, key_history, value_history):
        query_layer = self.transpose_for_scores(query)
        key_layer = self.transpose_for_scores(key)
        value_layer = self.transpose_for_scores(value)

        if (
            key_history is not None
            and not isinstance(key_history, list)
            and value_history is not None
        ):
            key_layer = torch.cat((key_history, key_layer), dim=-2)
            value_layer = torch.cat((value_history, value_layer), dim=-2)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # (batch, head, pos, pos)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask
        if rel_pos is not None:
            attention_scores = attention_scores + rel_pos

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # There was a debug implementation here, but I skipped.

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        if isinstance(key_history, list):
            key_history.append(key_layer)
        if isinstance(value_history, list):
            value_history.append(value_layer)

        return context_layer

class HBGLBertAttentionForDecoding(nn.Module):
    def __init__(self, config):
        super(HBGLBertAttentionForDecoding, self).__init__()
        self.self = HBGLBertSelfAttentionForDecoding(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, history_states=None,
                key_history=None, value_history=None, rel_pos=None):
        self_output = self.self(
            input_tensor, attention_mask, history_states=history_states,
            key_history=key_history, value_history=value_history, rel_pos=rel_pos)
        attention_output = self.output(self_output, input_tensor)
        return attention_output

class HBGLBertLayerForDecoding(nn.Module):
    def __init__(self, config):
        super(HBGLBertLayerForDecoding, self).__init__()
        self.attention = HBGLBertAttentionForDecoding(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor,
            history_states=None,
            key_history=None,
            value_history=None,
            rel_pos=None
    ):
        attention_output = self.attention(
            hidden_states, attention_mask, history_states=history_states,
            key_history=key_history, value_history=value_history, rel_pos=rel_pos)
        # log_variable("attention_output", attention_output)
        
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class HBGLBertEncoderForDecoding(nn.Module):
    def __init__(self, config):
        super(HBGLBertEncoderForDecoding, self).__init__()
        self.layer = nn.ModuleList([HBGLBertLayerForDecoding(config)
                                    for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True, 
                prev_embedding=None, prev_encoded_layers=None, 
                key_history=None, value_history=None, rel_pos=None):
        # history embedding and encoded layer must be simultanously given
        assert (prev_embedding is None) == (prev_encoded_layers is None)

        all_encoder_layers = []
        if (prev_embedding is not None) and (prev_encoded_layers is not None):
            history_states = prev_embedding
            for i, layer_module in enumerate(self.layer):
                hidden_states = layer_module(
                    hidden_states, attention_mask, history_states=history_states, 
                    rel_pos=rel_pos)
                if output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)
                if prev_encoded_layers is not None:
                    history_states = prev_encoded_layers[i]
        else:
            for i, layer_module in enumerate(self.layer):
                set_key = None
                if isinstance(key_history, list):
                    set_key = key_history if len(key_history) < len(self.layer) else key_history[i]
                set_value = None
                if isinstance(value_history, list):
                    assert key_history is not None
                    set_value = value_history if len(key_history) < len(self.layer) else value_history[i]
                hidden_states = layer_module(
                    hidden_states, attention_mask,
                    key_history=set_key, value_history=set_value, rel_pos=rel_pos)
                if output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class HBGLBertEmbeddingsForDecoding(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(HBGLBertEmbeddingsForDecoding, self).__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size)
        if config.type_vocab_size == 0:
            self.token_type_embeddings = None
        else:
            self.token_type_embeddings = nn.Embedding(
                config.type_vocab_size, config.hidden_size)
        if hasattr(config, 'fp32_embedding'):
            self.fp32_embedding = config.fp32_embedding
        else:
            self.fp32_embedding = False

        if hasattr(config, 'new_pos_ids') and config.new_pos_ids:
            self.num_pos_emb = 4
        else:
            self.num_pos_emb = 1
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size * self.num_pos_emb)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
            self,
            input_ids: torch.Tensor | None,
            token_type_ids: torch.Tensor | None = None,
            position_ids: torch.Tensor | None = None,
            task_idx: torch.Tensor | None = None,
            inputs_embeds: torch.Tensor | None = None
    ):
        if position_ids is None:
            raise NotImplementedError("Not handled yet")
        
        if token_type_ids is None:
            assert input_ids is not None
            token_type_ids = torch.zeros_like(input_ids)

        if input_ids is not None:
            words_embeddings: torch.Tensor = self.word_embeddings(input_ids)
        elif inputs_embeds is not None:
            words_embeddings = inputs_embeds
        else:
            raise ValueError("Either input_ids or input_embeds should not be None.")

        position_embeddings: torch.Tensor = self.position_embeddings(position_ids)

        if self.num_pos_emb > 1:
            num_batch = position_embeddings.size(0)
            num_pos = position_embeddings.size(1)
            position_embeddings = position_embeddings.view(
                num_batch, num_pos, self.num_pos_emb, -1)[torch.arange(0, num_batch).long(), :, task_idx, :]

        embeddings = words_embeddings + position_embeddings

        if self.token_type_embeddings is not None:
            embeddings: torch.Tensor = embeddings + self.token_type_embeddings(token_type_ids)

        if self.fp32_embedding:
            embeddings = embeddings.half()
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class HBGLBertModelForDecoding(BertPreTrainedModel):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").
    Params:
        config: a BertConfig class instance with the configuration to build a new model
    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.
    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLF`) to train on the Next-Sentence task (see BERT's paper).
    ```
    """

    def __init__(self, config):
        super(HBGLBertModelForDecoding, self).__init__(config)
        self.embeddings = HBGLBertEmbeddingsForDecoding(config)
        self.encoder = HBGLBertEncoderForDecoding(config)
        self.pooler = BertPooler(config)
        self.config = config

    def rescale_some_parameters(self):
        raise NotImplementedError("Is this even called?")

    def get_extended_attention_mask(self, input_ids, token_type_ids, attention_mask):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        #extended_attention_mask = extended_attention_mask.to(
            #dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True,
                task_idx=None, key_history=None, value_history=None, position_ids=None):
        raise NotImplementedError("It should be overrided by BertModelIncr")


class BertModelIncr(HBGLBertModelForDecoding):
    def __init__(self, config):
        super(BertModelIncr, self).__init__(config)
        self.pooler = None  # We don't care about the pooler.

        if self.config.rel_pos_bins > 0:
            self.rel_pos_bias = nn.Linear(self.config.rel_pos_bins, config.num_attention_heads, bias=False)
        else:
            self.rel_pos_bias = None

        self.post_init()

    def forward(self, input_ids, token_type_ids, position_ids, attention_mask, output_all_encoded_layers=True,
                prev_embedding=None, prev_encoded_layers=None, task_idx=None, rel_pos=None,
                inputs_embeds=None):
        extended_attention_mask = self.get_extended_attention_mask(
            input_ids, token_type_ids, attention_mask)

        embedding_output: torch.Tensor = self.embeddings(
            input_ids, token_type_ids, position_ids, task_idx=task_idx, inputs_embeds=inputs_embeds)

        if self.rel_pos_bias is not None and rel_pos is not None:
            # print("Rel pos size = %s" % str(rel_pos.size()))
            rel_pos = F.one_hot(rel_pos, num_classes=self.config.rel_pos_bins).type_as(embedding_output)
            # print("Rel pos size = %s" % str(rel_pos.size()))
            rel_pos = self.rel_pos_bias(rel_pos).permute(0, 3, 1, 2)
            # print("Rel pos size = %s" % str(rel_pos.size()))
        else:
            rel_pos = None
        encoded_layers: list[torch.Tensor] | torch.Tensor = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers,
                                      prev_embedding=prev_embedding,
                                      prev_encoded_layers=prev_encoded_layers,
                                      rel_pos=rel_pos)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return embedding_output, encoded_layers


class BertForSeq2SeqDecoder(BertPreTrainedModel):
    _tied_weights_keys = {
        "cls.predictions.decoder.weight": "bert.embeddings.word_embeddings.weight",
        "cls.predictions.decoder.bias": "cls.predictions.bias",
    }
    
    def __init__(self, config, mask_word_id=0, eos_id=0, pos_shift=False):
        super(BertForSeq2SeqDecoder, self).__init__(config)
        self.bert = BertModelIncr(config)
        self.cls = BertOnlyMLMHead(config)

        self.mask_word_id = mask_word_id
        self.eos_id = eos_id
        self.pos_shift = pos_shift

        self.label_start_index = -1
        self.ab_bound_token_id = -1
        self.hier_labels = None
        self.soft_label_hier_real = False

        self.post_init()

    def forward(
            self,
            input_ids: torch.Tensor,
            token_type_ids: torch.Tensor,
            position_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            task_idx: torch.Tensor | None = None,
    ):
        batch_size = input_ids.size(dim=0)
        input_length = input_ids.size(dim=1)
        output_length = token_type_ids.size(dim=1)

        output_ids = []
        prev_embedding = None
        prev_encoded_layers = None
        curr_ids = input_ids
        mask_ids = input_ids.new(batch_size, 1).fill_(self.mask_word_id)
        next_pos = input_length
        if self.pos_shift:
            sep_ids = input_ids.new(batch_size, 1).fill_(self.eos_id)

        if self.bert.rel_pos_bias is not None:
            rel_pos_mat = position_ids.unsqueeze(-2) - position_ids.unsqueeze(-1)
            rel_pos = relative_position_bucket(
                rel_pos_mat, num_buckets=self.config.rel_pos_bins, max_distance=self.config.max_rel_pos)
        else:
            rel_pos = None

        pred_embeds = None
        while next_pos < output_length:
            if curr_ids is None and pred_embeds is not None:
                # set curr_ids to None in two more loops
                curr_length = pred_embeds.size(dim=1)
                if self.pos_shift:
                    if next_pos == input_length:
                        x_input_embeds = torch.cat((pred_embeds,
                                                    self.bert.embeddings.word_embeddings(sep_ids)), dim=1)
                        start_pos = 0
                    else:
                        x_input_embeds = pred_embeds
                        start_pos = next_pos
                else:
                    start_pos = next_pos - curr_length
                    x_input_embeds = torch.cat((pred_embeds,
                                                self.bert.embeddings.word_embeddings(mask_ids)), dim=1)
                x_input_ids = None
            elif curr_ids is not None:
                curr_length = list(curr_ids.size())[1]
                if self.pos_shift:
                    if next_pos == input_length:
                        x_input_ids = torch.cat((curr_ids, sep_ids), dim=1)
                        start_pos = 0
                    else:
                        x_input_ids = curr_ids
                        start_pos = next_pos
                else:
                    start_pos = next_pos - curr_length
                    x_input_ids = torch.cat((curr_ids, mask_ids), dim=1)
                x_input_embeds = None
            else:
                raise ValueError("Cannot handle when curr_ids and pred_embeds are None")


            curr_token_type_ids = token_type_ids[:, start_pos:next_pos + 1]
            curr_attention_mask = attention_mask[:,
                                  start_pos:next_pos + 1, :next_pos + 1]
            curr_position_ids = position_ids[:, start_pos:next_pos + 1]

            if rel_pos is not None:
                cur_rel_pos = rel_pos[:, start_pos:next_pos + 1, :next_pos + 1]
            else:
                cur_rel_pos = None

            if x_input_embeds is not None:
                new_embedding, new_encoded_layers = \
                    self.bert(None, curr_token_type_ids, curr_position_ids, curr_attention_mask,
                            output_all_encoded_layers=True, prev_embedding=prev_embedding,
                              prev_encoded_layers=prev_encoded_layers, rel_pos=cur_rel_pos,
                              inputs_embeds=x_input_embeds, )
            else:
                new_embedding, new_encoded_layers = \
                    self.bert(x_input_ids, curr_token_type_ids, curr_position_ids, curr_attention_mask,
                            output_all_encoded_layers=True, prev_embedding=prev_embedding,
                            prev_encoded_layers=prev_encoded_layers, rel_pos=cur_rel_pos)
            assert isinstance(new_embedding, torch.Tensor)

            last_hidden = new_encoded_layers[-1][:, -1:, :]
            if self.soft_label_hier_real and self.hier_labels is not None:
                curr_hier = len(output_ids)
                hl = self.hier_labels[curr_hier]
                prediction_scores = self.cls(last_hidden)

                prediction_scores = torch.cat([prediction_scores[:, :, self.eos_id].unsqueeze(-1),
                                            prediction_scores[:, :, hl]], dim=-1)

                prediction_scores = torch.sigmoid(prediction_scores) > 0.5

                _pred_ids: list[torch.Tensor] = []
                for i in range(prediction_scores.shape[0]):
                    device = prediction_scores.device
                    pred_ids = torch.arange(prediction_scores.shape[-1], device=device)[prediction_scores[i, -1]]
                    sep_mask = pred_ids == 0
                    pred_ids[sep_mask] = self.eos_id
                    pred_ids[~sep_mask] = torch.arange(hl.shape[-1], device=device)[hl][pred_ids[~sep_mask] - 1]
                    _pred_ids.append(pred_ids)
                pred_ids = _pred_ids
                output_ids.append(pred_ids)

                pred_embeds =  prediction_scores.float() @ torch.cat(
                    [self.bert.embeddings.word_embeddings.weight.data[self.eos_id].unsqueeze(0),
                    self.bert.embeddings.word_embeddings.weight.data[hl]], dim=0)
                max_ids = None
            else:
                lsi = self.label_start_index
                prediction_scores = self.cls(last_hidden)

                prediction_scores = torch.cat([prediction_scores[:, :, self.eos_id].unsqueeze(-1),
                                            prediction_scores[:, :, lsi:]], dim=-1)

                prediction_scores = torch.sigmoid(prediction_scores) > 0.5

                _pred_ids: list[torch.Tensor] = []
                for i in range(prediction_scores.shape[0]):
                    device = prediction_scores.device
                    pred_ids = torch.arange(prediction_scores.shape[-1], device=device)[prediction_scores[i, -1]]
                    sep_mask = pred_ids == 0
                    pred_ids[sep_mask] = self.eos_id
                    pred_ids[~sep_mask] += lsi - 1
                    _pred_ids.append(pred_ids)
                pred_ids = _pred_ids
                output_ids.append(pred_ids)

                pred_embeds =  prediction_scores.float() @ torch.cat(
                    [self.bert.embeddings.word_embeddings.weight.data[self.eos_id].unsqueeze(0),
                    self.bert.embeddings.word_embeddings.weight.data[lsi:]], dim=0)
                max_ids = None

            if self.pos_shift:
                if prev_embedding is None:
                    prev_embedding = new_embedding
                else:
                    prev_embedding = torch.cat(
                        (prev_embedding, new_embedding), dim=1)
                if prev_encoded_layers is None:
                    prev_encoded_layers = [x for x in new_encoded_layers]
                else:
                    prev_encoded_layers = [torch.cat((x[0], x[1]), dim=1) for x in zip(
                        prev_encoded_layers, new_encoded_layers)]
            else:
                if prev_embedding is None:
                    prev_embedding = new_embedding[:, :-1, :]
                else:
                    prev_embedding = torch.cat(
                        (prev_embedding, new_embedding[:, :-1, :]), dim=1)
                if prev_encoded_layers is None:
                    prev_encoded_layers = [x[:, :-1, :]
                                           for x in new_encoded_layers]
                else:
                    prev_encoded_layers = [torch.cat((x[0], x[1][:, :-1, :]), dim=1)
                                           for x in zip(prev_encoded_layers, new_encoded_layers)]
            curr_ids = max_ids
            next_pos += 1

        device = output_ids[0][0].device
        _output_ids: list = [[] for _ in output_ids[0]]
        max_l = 0
        for i, oi in enumerate(_output_ids):
            
            _output_ids[i] = torch.cat([j[i] for j in output_ids], dim=0)
            index = 10000
            for index, d in enumerate(_output_ids[i].tolist()):
                if d == 102: break
            _output_ids[i] = sorted(list(set(_output_ids[i].tolist()[:index])), reverse=True)
            max_l = max(len(_output_ids[i]), max_l)

        for i, oi in enumerate(_output_ids):
            _output_ids[i] = torch.tensor(oi + [0] * (max_l - len(oi)), dtype=torch.long, device=device).unsqueeze(0)
        output_ids = _output_ids
        return torch.cat(output_ids, dim=0)
