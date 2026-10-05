from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import math

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
from transformers import BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder, BertOnlyMLMHead, BertPooler

from config import BertForSeq2SeqConfig

logger = logging.getLogger(__name__)

BertLayerNorm = torch.nn.LayerNorm

class BertPreTrainedForSeq2SeqModel(BertPreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    config_class = BertForSeq2SeqConfig
    base_model_prefix = "unilm_for_seq2seq"

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, reuse_position_embedding=None, *model_args, **kwargs):
        temp_model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        config = kwargs["config"]
        state_dict = temp_model.state_dict()

        _k = 'bert.embeddings.position_embeddings.weight'
        if _k in state_dict:
            if config.max_position_embeddings > state_dict[_k].shape[0]:
                logger.info("Resize > position embeddings !")
                old_vocab_size = state_dict[_k].shape[0]
                new_postion_embedding = state_dict[_k].data.new_tensor(torch.ones(
                    size=(config.max_position_embeddings, state_dict[_k].shape[1])), dtype=torch.float)
                new_postion_embedding = nn.Parameter(data=new_postion_embedding, requires_grad=True)
                new_postion_embedding.data.normal_(mean=0.0, std=config.initializer_range)
                max_range = config.max_position_embeddings if reuse_position_embedding else old_vocab_size
                shift = 0
                while shift < max_range:
                    delta = min(old_vocab_size, max_range - shift)
                    new_postion_embedding.data[shift: shift + delta, :] = state_dict[_k][:delta, :]
                    logger.info("  CP [%d ~ %d] into [%d ~ %d]  " % (0, delta, shift, shift + delta))
                    shift += delta
                state_dict[_k] = new_postion_embedding.data
                del new_postion_embedding
            elif config.max_position_embeddings < state_dict[_k].shape[0]:
                logger.info("Resize < position embeddings !")
                old_vocab_size = state_dict[_k].shape[0]
                new_postion_embedding = state_dict[_k].data.new_tensor(torch.ones(
                    size=(config.max_position_embeddings, state_dict[_k].shape[1])), dtype=torch.float)
                new_postion_embedding = nn.Parameter(data=new_postion_embedding, requires_grad=True)
                new_postion_embedding.data.normal_(mean=0.0, std=config.initializer_range)
                new_postion_embedding.data.copy_(state_dict[_k][:config.max_position_embeddings, :])
                state_dict[_k] = new_postion_embedding.data
                del new_postion_embedding

        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

def relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
    """
    Adapted from Mesh Tensorflow:
    https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593
    """
    ret = 0
    if bidirectional:
        num_buckets //= 2
        # mtf.to_int32(mtf.less(n, 0)) * num_buckets
        ret += (relative_position > 0).long() * num_buckets
        n = torch.abs(relative_position)
    else:
        n = torch.max(-relative_position, torch.zeros_like(relative_position))
    # now n is in the range [0, inf)

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = n < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    val_if_large = max_exact + (
        torch.log(n.float() / max_exact) / math.log(max_distance /
                                                    max_exact) * (num_buckets - max_exact)
    ).to(torch.long)
    val_if_large = torch.min(
        val_if_large, torch.full_like(val_if_large, num_buckets - 1))

    ret += torch.where(is_small, n, val_if_large)
    return ret


class BertModel(BertPreTrainedForSeq2SeqModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the output of the last layer of the model.
        **pooler_output**: ``torch.FloatTensor`` of shape ``(batch_size, hidden_size)``
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during Bert pretraining. This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

    """
    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        if not isinstance(config, BertForSeq2SeqConfig):
            self.pooler = BertPooler(config)
        else:
            self.pooler = None

        if self.config.rel_pos_bins > 0:
            self.rel_pos_bias = nn.Linear(self.config.rel_pos_bins, config.num_attention_heads, bias=False)
        else:
            self.rel_pos_bias = None

        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, inputs_embeds=None, split_lengths=None):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            device = input_ids.device
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            device = inputs_embeds.device
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)

        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask=attention_mask,
            input_shape=input_shape
        )

        embedding_output, position_ids = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds)
        if self.config.rel_pos_bins > 0 and self.rel_pos_bias is not None:
            rel_pos_mat = position_ids.unsqueeze(-2) - position_ids.unsqueeze(-1)
            rel_pos = relative_position_bucket(
                rel_pos_mat, num_buckets=self.config.rel_pos_bins, max_distance=self.config.max_rel_pos)
            rel_pos = F.one_hot(rel_pos, num_classes=self.config.rel_pos_bins).type_as(embedding_output)
            rel_pos = self.rel_pos_bias(rel_pos).permute(0, 3, 1, 2)
        else:
            rel_pos = None
        encoder_outputs = self.encoder(
            embedding_output, attention_mask=extended_attention_mask, 
            split_lengths=split_lengths, rel_pos=rel_pos)
        sequence_output = encoder_outputs[0]

        outputs = (sequence_output, ) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
        if self.pooler is None:
            return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)
        else:
            pooled_output = self.pooler(sequence_output)
            return sequence_output, pooled_output


class LabelSmoothingLoss(_Loss):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """

    def __init__(self, label_smoothing=0, tgt_vocab_size=0, ignore_index=0, size_average=None, reduce=None, reduction='mean'):
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__(
            size_average=size_average, reduce=reduce, reduction=reduction)

        assert label_smoothing > 0
        assert tgt_vocab_size > 0

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.ignore_index] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))
        self.confidence = 1.0 - label_smoothing
        self.tgt_vocab_size = tgt_vocab_size

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size * num_pos * n_classes
        target (LongTensor): batch_size * num_pos
        """
        assert self.tgt_vocab_size == output.size(2)
        batch_size, num_pos = target.size(0), target.size(1)
        output = output.view(-1, self.tgt_vocab_size)
        target = target.view(-1)

        model_prob = self.one_hot.float()
        if not isinstance(model_prob, torch.Tensor):
            raise ValueError("Expecting tensor in model_prob")

        model_prob = model_prob.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

        return F.kl_div(output, model_prob, reduction='none').view(batch_size, num_pos, -1).sum(2)


def create_mask_and_position_ids(num_tokens, max_len, offset=None):
    base_position_matrix = torch.arange(
        0, max_len, dtype=num_tokens.dtype, device=num_tokens.device).view(1, -1)
    mask = (base_position_matrix < num_tokens.view(-1, 1)).type_as(num_tokens)
    if offset is not None:
        base_position_matrix = base_position_matrix + offset.view(-1, 1)
    position_ids = base_position_matrix * mask
    return mask, position_ids


class BertForSequenceToSequence(BertPreTrainedForSeq2SeqModel):
    MODEL_NAME = 'basic class'
    soft_label: bool = False
    mask_token_id: int = -1
    sep_token_id: int = -1
    vs: int = -1

    def __init__(self, config):
        super(BertForSequenceToSequence, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)
        self.cls.predictions.decoder.weight = self.bert.embeddings.word_embeddings.weight
        self.init_weights()

        self.log_softmax = nn.LogSoftmax()

        self.source_type_id = config.source_type_id
        self.target_type_id = config.target_type_id

        if config.label_smoothing > 0:
            self.crit_mask_lm_smoothed = LabelSmoothingLoss(
                config.label_smoothing, config.vocab_size, ignore_index=0, reduction='none')
            self.crit_mask_lm = None
        else:
            self.crit_mask_lm_smoothed = None
            self.crit_mask_lm = nn.CrossEntropyLoss(reduction='none')

        self.post_init()


class BertForSequenceToSequenceWithPseudoMask(BertForSequenceToSequence):
    label_start_index = -1
    ab_bound_token_id = -1
    MODEL_NAME = "BertForSequenceToSequenceWithPseudoMask"
    soft_label_hier_real = False

    @staticmethod
    def create_attention_mask(source_mask, target_mask, source_position_ids, target_span_ids):
        weight = torch.cat((torch.zeros_like(source_position_ids), target_span_ids, -target_span_ids), dim=1)
        from_weight = weight.unsqueeze(-1)
        to_weight = weight.unsqueeze(1)

        true_tokens = (0 <= to_weight) & (torch.cat((source_mask, target_mask, target_mask), dim=1) == 1).unsqueeze(1)
        true_tokens_mask = (from_weight >= 0) & true_tokens & (to_weight <= from_weight)
        pseudo_tokens_mask = (from_weight < 0) & true_tokens & (-to_weight > from_weight)
        pseudo_tokens_mask = pseudo_tokens_mask | ((from_weight < 0) & (to_weight == from_weight))

        return (true_tokens_mask | pseudo_tokens_mask).type_as(source_mask)

    def forward(
            self, source_ids, target_ids, label_ids, pseudo_ids, 
            num_source_tokens, num_target_tokens, target_span_ids=None, target_no_offset=None):
        source_len = source_ids.size(1)
        target_len = target_ids.size(1)
        pseudo_len = pseudo_ids.size(1)
        assert target_len == pseudo_len
        assert source_len > 0 and target_len > 0
        split_lengths = (source_len, target_len, pseudo_len)


        if self.soft_label:
            # input_ids = torch.cat((source_ids, target_ids, pseudo_ids), dim=1)
            source_ids_embeds = self.bert.embeddings.word_embeddings(source_ids)

            target_ids_embeds = target_ids @ self.bert.embeddings.word_embeddings.weight.data
            pseudo_ids_embeds = self.bert.embeddings.word_embeddings(pseudo_ids)
            input_embeds = torch.cat((source_ids_embeds, target_ids_embeds, pseudo_ids_embeds), dim=1)
            input_ids = None
            assert pseudo_ids.shape[1] ==  target_ids.shape[1]
            token_type_ids = torch.cat(
                (torch.ones_like(source_ids) * self.source_type_id,
                 torch.ones_like(pseudo_ids) * self.target_type_id,
                 torch.ones_like(pseudo_ids) * self.target_type_id), dim=1)
        else:
            input_embeds = None
            input_ids = torch.cat((source_ids, target_ids, pseudo_ids), dim=1)
            token_type_ids = torch.cat(
                (torch.ones_like(source_ids) * self.source_type_id,
                 torch.ones_like(target_ids) * self.target_type_id,
                 torch.ones_like(pseudo_ids) * self.target_type_id), dim=1)

        source_mask, source_position_ids = \
            create_mask_and_position_ids(num_source_tokens, source_len)
        target_mask, target_position_ids = \
            create_mask_and_position_ids(
                num_target_tokens, target_len, offset=None if target_no_offset else num_source_tokens)

        position_ids = torch.cat((source_position_ids, target_position_ids, target_position_ids), dim=1)
        if target_span_ids is None:
            target_span_ids = target_position_ids
        attention_mask = self.create_attention_mask(source_mask, target_mask, source_position_ids, target_span_ids)

        if self.soft_label:
            outputs = self.bert(
                None, inputs_embeds=input_embeds, attention_mask=attention_mask, token_type_ids=token_type_ids,
                position_ids=position_ids, split_lengths=split_lengths)
        else:
            outputs = self.bert(
                input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                position_ids=position_ids, split_lengths=split_lengths)

        sequence_output = outputs[0]
        pseudo_sequence_output = sequence_output[:, source_len + target_len:, ]

        def loss_mask_and_normalize(loss, mask):
            mask = mask.type_as(loss)
            loss = loss * mask
            denominator = torch.sum(mask) + 1e-5
            return (loss / denominator).sum()

        if self.label_start_index > 0:
            lsi = self.label_start_index
            new_label_ids = label_ids - lsi
            new_label_ids += 1

            sep_tokens = list(set(label_ids[new_label_ids < 0].cpu().tolist()))
            assert len(sep_tokens) == 1
            sep_token = sep_tokens[0]

            new_label_ids[new_label_ids < 0] = 0 # sep token

            prediction_scores_masked = self.cls(pseudo_sequence_output)
            prediction_scores_masked = torch.cat([prediction_scores_masked[:, :, sep_token].unsqueeze(-1),
                                                  prediction_scores_masked[:, :, lsi:]], dim=-1)

            label_ids = new_label_ids
            # print(prediction_scores_masked, label_ids)
        elif self.ab_bound_token_id > 0:
            # prediction_scores_masked = self.cls(pseudo_sequence_output)
            hidden_states = self.cls.predictions.transform(pseudo_sequence_output)
            prediction_scores_masked = F.linear(hidden_states, weight=self.cls.predictions.decoder.weight, bias=self.cls.predictions.bias)

            b_token_embs = self.cls.predictions.decoder.weight[self.ab_bound_token_id:, ]
            new_b_token_embs = []
            new_b_hidden_states  = []
            for i, label_id in enumerate(label_ids.tolist()):
                a = -1
                for j, l in enumerate(label_id):
                    if l == 102 or l == 0: break
                    if l > self.ab_bound_token_id:
                        assert a > 0
                        new_b_token_embs.append(b_token_embs + self.cls.predictions.decoder.weight[a])
                        new_b_hidden_states.append(hidden_states[i, j])
                    else:
                        a = l
            new_b_hidden_states = torch.stack(new_b_hidden_states, dim=0)
            new_b_token_embs = torch.stack(new_b_token_embs, dim=0)
            prediction_scores_masked[label_ids > self.ab_bound_token_id][:,self.ab_bound_token_id:] =\
                torch.bmm(new_b_token_embs, new_b_hidden_states.unsqueeze(-1)).squeeze(-1) + self.cls.predictions.bias[self.ab_bound_token_id:]
        else:
            prediction_scores_masked = self.cls(pseudo_sequence_output)

        if self.soft_label and isinstance(self.sep_token_id, int):
            if (
                self.soft_label_hier_real
                and isinstance(self.hier_labels, torch.Tensor)
            ):
                loss_fct = BCEWithLogitsLoss()
                pseudo_lm_loss = 0
                for i, hl in enumerate(self.hier_labels):
                    mask_token_mask = pseudo_ids[:, i] == self.mask_token_id
                    if mask_token_mask.sum().item() == 0:
                        continue
                    hl = hl.to(pseudo_ids.device)
                    prediction_scores_masked_i = prediction_scores_masked[:, i][mask_token_mask]
                    label_ids_i = label_ids[:, i][mask_token_mask]
                    prediction_scores_masked_i = torch.cat([prediction_scores_masked_i[:, self.sep_token_id].unsqueeze(-1),
                                                            prediction_scores_masked_i[:, hl]], dim=-1)
                    label_ids_i = torch.cat([label_ids_i[:, self.sep_token_id].unsqueeze(-1),
                                             label_ids_i[:, hl]], dim=-1)
                    pseudo_lm_loss += loss_fct(prediction_scores_masked_i, label_ids_i)

                # sep token
                sep_mask_token_mask = pseudo_ids[:, -1] == self.mask_token_id
                if sep_mask_token_mask.sum().item() != 0:
                    prediction_scores_masked_sep = prediction_scores_masked[:, -1][sep_mask_token_mask]
                    label_ids_sep = label_ids[:, -1][sep_mask_token_mask]
                    prediction_scores_masked_sep = torch.cat([prediction_scores_masked_sep[:, self.sep_token_id].unsqueeze(-1),
                                                    prediction_scores_masked_sep[:, self.vs:]], dim=-1)
                    label_ids_sep = torch.cat([label_ids_sep[:, self.sep_token_id].unsqueeze(-1),
                                               label_ids_sep[:, self.vs:]], dim=-1)
                    pseudo_lm_loss += loss_fct(prediction_scores_masked, label_ids)
            else:
                loss_fct = BCEWithLogitsLoss()
                mask_token_mask = pseudo_ids == self.mask_token_id
                prediction_scores_masked = prediction_scores_masked[mask_token_mask]
                label_ids = label_ids[mask_token_mask]

                # label_ids[label_ids != self.sep_token_id] += 1

                prediction_scores_masked = torch.cat([prediction_scores_masked[:, self.sep_token_id].unsqueeze(-1),
                                                    prediction_scores_masked[:, self.vs:]], dim=-1)
                label_ids = torch.cat([label_ids[:, self.sep_token_id].unsqueeze(-1),
                                    label_ids[:, self.vs:]], dim=-1)
                pseudo_lm_loss = loss_fct(prediction_scores_masked, label_ids)

                # masked_lm_loss = loss_fct(prediction_scores_masked[mask_token_mask], label_ids[mask_token_mask])
        elif self.crit_mask_lm_smoothed:
            masked_lm_loss = self.crit_mask_lm_smoothed(
                F.log_softmax(prediction_scores_masked.float(), dim=-1), label_ids)
            pseudo_lm_loss = loss_mask_and_normalize(
                masked_lm_loss.float(), target_mask)
        elif self.crit_mask_lm:
            masked_lm_loss = self.crit_mask_lm(
                prediction_scores_masked.transpose(1, 2).float(), label_ids)
            pseudo_lm_loss = loss_mask_and_normalize(
                masked_lm_loss.float(), target_mask)
        else:
            raise NotImplementedError("No loss implemented")

        return pseudo_lm_loss


# class BertForSequenceToSequenceUniLMV1(BertForSequenceToSequence):
#     MODEL_NAME = "BertForSequenceToSequenceUniLMV1"

#     @staticmethod
#     def create_attention_mask(source_mask, target_mask, source_position_ids, target_span_ids):
#         weight = torch.cat((torch.zeros_like(source_position_ids), target_span_ids), dim=1)
#         from_weight = weight.unsqueeze(-1)
#         to_weight = weight.unsqueeze(1)

#         true_tokens = torch.cat((source_mask, target_mask), dim=1).unsqueeze(1)
#         return ((true_tokens == 1) & (to_weight <= from_weight)).type_as(source_mask)

#     def forward(self, source_ids, target_ids, masked_ids, masked_pos, masked_weight, num_source_tokens, num_target_tokens):
#         source_len = source_ids.size(1)
#         target_len = target_ids.size(1)
#         split_lengths = (source_len, target_len)

#         input_ids = torch.cat((source_ids, target_ids), dim=1)

#         token_type_ids = torch.cat(
#             (torch.ones_like(source_ids) * self.source_type_id,
#              torch.ones_like(target_ids) * self.target_type_id), dim=1)

#         source_mask, source_position_ids = \
#             create_mask_and_position_ids(num_source_tokens, source_len)
#         target_mask, target_position_ids = \
#             create_mask_and_position_ids(
#                 num_target_tokens, target_len, offset=num_source_tokens)

#         position_ids = torch.cat((source_position_ids, target_position_ids), dim=1)
#         attention_mask = self.create_attention_mask(
#             source_mask, target_mask, source_position_ids, target_position_ids)

#         outputs = self.bert(
#             input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
#             position_ids=position_ids, split_lengths=split_lengths)

#         def gather_seq_out_by_pos(seq, pos):
#             return torch.gather(seq, 1, pos.unsqueeze(2).expand(-1, -1, seq.size(-1)))
        
#         sequence_output = outputs[0]
#         target_sequence_output = sequence_output[:, source_len:, ]
#         masked_sequence_output = gather_seq_out_by_pos(target_sequence_output, masked_pos)

#         def loss_mask_and_normalize(loss, mask):
#             mask = mask.type_as(loss)
#             loss = loss * mask
#             denominator = torch.sum(mask) + 1e-5
#             return (loss / denominator).sum()

#         prediction_scores_masked = self.cls(masked_sequence_output)

#         if self.crit_mask_lm_smoothed:
#             masked_lm_loss = self.crit_mask_lm_smoothed(
#                 F.log_softmax(prediction_scores_masked.float(), dim=-1), masked_ids)
#         elif self.crit_mask_lm:
#             masked_lm_loss = self.crit_mask_lm(
#                 prediction_scores_masked.transpose(1, 2).float(), masked_ids)
#         else:
#             raise NotImplementedError("No loss implemented")
#         pseudo_lm_loss = loss_mask_and_normalize(
#             masked_lm_loss.float(), masked_weight)

#         return pseudo_lm_loss
