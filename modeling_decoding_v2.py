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
from transformers.models.bert.modeling_bert import BertModel, BertOnlyMLMHead, BertPreTrainedModel

from modeling import relative_position_bucket

logger = logging.getLogger(__name__)


class BertModelIncr(BertModel):
    def __init__(self, config):
        super(BertModelIncr, self).__init__(config)
        self.pooler = None  # We don't care about the pooler.

        if self.config.rel_pos_bins > 0:
            self.rel_pos_bias = nn.Linear(self.config.rel_pos_bins, config.num_attention_heads, bias=False)
        else:
            self.rel_pos_bias = None

        self.post_init()

    def forward(self, input_ids, token_type_ids, position_ids, attention_mask, output_all_encoded_layers=True,
                prev_embedding=None, prev_encoded_layers=None, mask_qkv=None, task_idx=None, rel_pos=None,
                inputs_embeds=None):
        extended_attention_mask = self.get_extended_attention_mask(
            input_ids, token_type_ids, attention_mask)

        embedding_output = self.embeddings(
            input_ids, token_type_ids, position_ids, task_idx=task_idx, inputs_embeds=inputs_embeds)

        if self.rel_pos_bias is not None and rel_pos is not None:
            # print("Rel pos size = %s" % str(rel_pos.size()))
            rel_pos = F.one_hot(rel_pos, num_classes=self.config.rel_pos_bins).type_as(embedding_output)
            # print("Rel pos size = %s" % str(rel_pos.size()))
            rel_pos = self.rel_pos_bias(rel_pos).permute(0, 3, 1, 2)
            # print("Rel pos size = %s" % str(rel_pos.size()))
        else:
            rel_pos = None
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers,
                                      prev_embedding=prev_embedding,
                                      prev_encoded_layers=prev_encoded_layers, mask_qkv=mask_qkv,
                                      seg_ids=token_type_ids, rel_pos=rel_pos)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return embedding_output, encoded_layers


class BertForSeq2SeqDecoder(BertPreTrainedModel):
    _tied_weights_keys = {
        "cls.predictions.decoder.weight": "bert.embeddings.word_embeddings.weight",
        "cls.predictions.decoder.bias": "cls.predictions.bias",
    }
    
    def __init__(self, config, mask_word_id=0, num_labels=2, num_rel=0,
                 search_beam_size=1, length_penalty=1.0, eos_id=0, sos_id=0,
                 forbid_duplicate_ngrams=False, forbid_ignore_set=None, ngram_size=3, min_len=0, mode="s2s",
                 pos_shift=False):
        super(BertForSeq2SeqDecoder, self).__init__(config)
        self.bert = BertModelIncr(config)
        self.cls = BertOnlyMLMHead(config)

        self.crit_mask_lm = nn.CrossEntropyLoss(reduction='none')
        self.crit_next_sent = nn.CrossEntropyLoss(ignore_index=-1)
        self.mask_word_id = mask_word_id
        self.num_labels = num_labels
        self.search_beam_size = search_beam_size
        self.length_penalty = length_penalty
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.forbid_duplicate_ngrams = forbid_duplicate_ngrams
        self.forbid_ignore_set = forbid_ignore_set
        self.ngram_size = ngram_size
        self.min_len = min_len
        assert mode in ("s2s", "l2r")
        self.mode = mode
        self.pos_shift = pos_shift

        self.label_start_index = -1
        self.ab_bound_token_id = -1
        self.soft_label = False
        self.hier_labels = None
        self.soft_label_hier_real = False

        self.post_init()

    def forward(self, input_ids, token_type_ids, position_ids, attention_mask, task_idx=None, mask_qkv=None):
        if self.search_beam_size > 1:
            return self.beam_search(input_ids, token_type_ids, position_ids, attention_mask,
                                    task_idx=task_idx, mask_qkv=mask_qkv)

        input_shape = list(input_ids.size())
        batch_size = input_shape[0]
        input_length = input_shape[1]
        output_shape = list(token_type_ids.size())
        output_length = output_shape[1]

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
            if self.soft_label and curr_ids is None and pred_embeds is not None:
                # set curr_ids to None in two more loops
                curr_length = pred_embeds.size()[1]
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
                raise ValueError("curr_ids should not None or soft_label is active")


            curr_token_type_ids = token_type_ids[:, start_pos:next_pos + 1]
            curr_attention_mask = attention_mask[:,
                                  start_pos:next_pos + 1, :next_pos + 1]
            curr_position_ids = position_ids[:, start_pos:next_pos + 1]

            if rel_pos is not None:
                cur_rel_pos = rel_pos[:, start_pos:next_pos + 1, :next_pos + 1]
            else:
                cur_rel_pos = None

            if self.soft_label and x_input_embeds is not None:
                new_embedding, new_encoded_layers = \
                    self.bert(None, curr_token_type_ids, curr_position_ids, curr_attention_mask,
                            output_all_encoded_layers=True, prev_embedding=prev_embedding,
                              prev_encoded_layers=prev_encoded_layers, mask_qkv=mask_qkv, rel_pos=cur_rel_pos,
                              inputs_embeds=x_input_embeds, )
            else:
                new_embedding, new_encoded_layers = \
                    self.bert(x_input_ids, curr_token_type_ids, curr_position_ids, curr_attention_mask,
                            output_all_encoded_layers=True, prev_embedding=prev_embedding,
                            prev_encoded_layers=prev_encoded_layers, mask_qkv=mask_qkv, rel_pos=cur_rel_pos)

            last_hidden = new_encoded_layers[-1][:, -1:, :]
            if self.soft_label:
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
            elif self.label_start_index > 0:
                lsi = self.label_start_index
                prediction_scores = self.cls(last_hidden)

                prediction_scores = torch.cat([prediction_scores[:, :, self.eos_id].unsqueeze(-1),
                                               prediction_scores[:, :, lsi:]], dim=-1)

                _, max_ids = torch.max(prediction_scores, dim=-1)
                sep_mask = max_ids == 0
                max_ids[sep_mask] = self.eos_id
                max_ids[~sep_mask] += lsi - 1
                output_ids.append(max_ids)
            elif self.ab_bound_token_id > 0:
                if len(output_ids) > 0:
                    last_output_ids = output_ids[-1]
                    hidden_states = self.cls.predictions.transform(last_hidden)
                    prediction_scores = F.linear(hidden_states, weight=self.cls.predictions.decoder.weight, bias=self.cls.predictions.bias)
                    b_token_embs = self.cls.predictions.decoder.weight[self.ab_bound_token_id:, ]

                    for i, label_id in enumerate(last_output_ids.view(-1).tolist()):
                        if label_id < self.ab_bound_token_id:
                            new_b_token_embs = b_token_embs + self.cls.predictions.decoder.weight[label_id]
                            new_b_scores = F.linear(hidden_states[i], weight=new_b_token_embs,
                                                    bias=self.cls.predictions.bias[self.ab_bound_token_id:])
                            prediction_scores[i, :,   self.ab_bound_token_id:] = new_b_scores

                    _, max_ids = torch.max(prediction_scores, dim=-1)
                    output_ids.append(max_ids)
                else:
                    prediction_scores = self.cls(last_hidden)
                    _, max_ids = torch.max(prediction_scores, dim=-1)
                    output_ids.append(max_ids)
            else:
                prediction_scores = self.cls(last_hidden)
                _, max_ids = torch.max(prediction_scores, dim=-1)
                output_ids.append(max_ids)

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

        if self.soft_label:
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
        else:
            return torch.cat(output_ids, dim=1)

    def beam_search(self, input_ids, token_type_ids, position_ids, attention_mask, task_idx=None, mask_qkv=None):
        input_shape = list(input_ids.size())
        batch_size = input_shape[0]
        input_length = input_shape[1]
        output_shape = list(token_type_ids.size())
        output_length = output_shape[1]

        output_ids = []
        prev_embedding = None
        prev_encoded_layers = None
        curr_ids = input_ids
        mask_ids = input_ids.new(batch_size, 1).fill_(self.mask_word_id)
        next_pos = input_length
        if self.pos_shift:
            sep_ids = input_ids.new(batch_size, 1).fill_(self.eos_id)

        K = self.search_beam_size

        total_scores = []
        beam_masks = []
        step_ids = []
        step_back_ptrs = []
        partial_seqs = []
        forbid_word_mask = None
        buf_matrix = None

        if self.bert.rel_pos_bias is not None:
            rel_pos_mat = position_ids.unsqueeze(-2) - position_ids.unsqueeze(-1)
            rel_pos = relative_position_bucket(
                rel_pos_mat, num_buckets=self.config.rel_pos_bins, max_distance=self.config.max_rel_pos)
        else:
            rel_pos = None
        # print("Rel pos size = %s" % str(rel_pos.size()))

        while next_pos < output_length:
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

            curr_token_type_ids = token_type_ids[:, start_pos:next_pos + 1]
            curr_attention_mask = attention_mask[:, start_pos:next_pos + 1, :next_pos + 1]
            curr_position_ids = position_ids[:, start_pos:next_pos + 1]
            if rel_pos is not None:
                cur_rel_pos = rel_pos[:, start_pos:next_pos + 1, :next_pos + 1]
            else:
                cur_rel_pos = None
            new_embedding, new_encoded_layers = \
                self.bert(x_input_ids, curr_token_type_ids, curr_position_ids, curr_attention_mask,
                          output_all_encoded_layers=True, prev_embedding=prev_embedding,
                          prev_encoded_layers=prev_encoded_layers, mask_qkv=mask_qkv, rel_pos=cur_rel_pos)

            last_hidden = new_encoded_layers[-1][:, -1:, :]
            prediction_scores = self.cls(last_hidden)
            log_scores = torch.nn.functional.log_softmax(
                prediction_scores, dim=-1)
            
            if forbid_word_mask is not None:
                log_scores += (forbid_word_mask * -10000.0)
            if self.min_len and (next_pos - input_length + 1 <= self.min_len):
                log_scores[:, :, self.eos_id].fill_(-10000.0)
            kk_scores, kk_ids = torch.topk(log_scores, k=K)
            if len(total_scores) == 0:
                k_ids = torch.reshape(kk_ids, [batch_size, K])
                back_ptrs = torch.zeros(batch_size, K, dtype=torch.long)
                k_scores = torch.reshape(kk_scores, [batch_size, K])
            else:
                last_eos = torch.reshape(
                    beam_masks[-1], [batch_size * K, 1, 1])
                last_seq_scores = torch.reshape(
                    total_scores[-1], [batch_size * K, 1, 1])
                kk_scores += last_eos * (-10000.0) + last_seq_scores
                kk_scores = torch.reshape(kk_scores, [batch_size, K * K])
                k_scores, k_ids = torch.topk(kk_scores, k=K)
                back_ptrs = torch.div(k_ids, K)
                back_ptrs = back_ptrs.long()
                kk_ids = torch.reshape(kk_ids, [batch_size, K * K])
                k_ids = torch.gather(kk_ids, 1, k_ids)
            step_back_ptrs.append(back_ptrs)
            step_ids.append(k_ids)
            beam_masks.append(torch.eq(k_ids, self.eos_id).type_as(kk_scores))
            total_scores.append(k_scores)

            def first_expand(x):
                input_shape = list(x.size())
                expanded_shape = input_shape[:1] + [1] + input_shape[1:]
                x = torch.reshape(x, expanded_shape)
                repeat_count = [1, K] + [1] * (len(input_shape) - 1)
                x = x.repeat(*repeat_count)
                x = torch.reshape(x, [input_shape[0] * K] + input_shape[1:])
                return x

            def select_beam_items(x, ids):
                id_shape = list(ids.size())
                id_rank = len(id_shape)
                assert len(id_shape) == 2
                x_shape = list(x.size())
                x = torch.reshape(x, [batch_size, K] + x_shape[1:])
                x_rank = len(x_shape) + 1
                assert x_rank >= 2
                if id_rank < x_rank:
                    ids = torch.reshape(
                        ids, id_shape + [1] * (x_rank - id_rank))
                    ids = ids.expand(id_shape + x_shape[1:])
                y = torch.gather(x, 1, ids)
                y = torch.reshape(y, x_shape)
                return y

            is_first = (prev_embedding is None)

            if self.pos_shift:
                if prev_embedding is None:
                    prev_embedding = first_expand(new_embedding)
                else:
                    prev_embedding = torch.cat(
                        (prev_embedding, new_embedding), dim=1)
                    prev_embedding = select_beam_items(
                        prev_embedding, back_ptrs)
                if prev_encoded_layers is None:
                    prev_encoded_layers = [first_expand(
                        x) for x in new_encoded_layers]
                else:
                    prev_encoded_layers = [torch.cat((x[0], x[1]), dim=1) for x in zip(
                        prev_encoded_layers, new_encoded_layers)]
                    prev_encoded_layers = [select_beam_items(
                        x, back_ptrs) for x in prev_encoded_layers]
            else:
                if prev_embedding is None:
                    prev_embedding = first_expand(new_embedding[:, :-1, :])
                else:
                    prev_embedding = torch.cat(
                        (prev_embedding, new_embedding[:, :-1, :]), dim=1)
                    prev_embedding = select_beam_items(
                        prev_embedding, back_ptrs)
                if prev_encoded_layers is None:
                    prev_encoded_layers = [first_expand(
                        x[:, :-1, :]) for x in new_encoded_layers]
                else:
                    prev_encoded_layers = [torch.cat((x[0], x[1][:, :-1, :]), dim=1)
                                           for x in zip(prev_encoded_layers, new_encoded_layers)]
                    prev_encoded_layers = [select_beam_items(
                        x, back_ptrs) for x in prev_encoded_layers]

            curr_ids = torch.reshape(k_ids, [batch_size * K, 1])

            if is_first:
                token_type_ids = first_expand(token_type_ids)
                position_ids = first_expand(position_ids)
                attention_mask = first_expand(attention_mask)
                if rel_pos is not None:
                    rel_pos = first_expand(rel_pos)
                mask_ids = first_expand(mask_ids)
                if mask_qkv is not None:
                    mask_qkv = first_expand(mask_qkv)

            if self.forbid_duplicate_ngrams:
                wids = step_ids[-1].tolist()
                ptrs = step_back_ptrs[-1].tolist()
                if is_first:
                    partial_seqs = []
                    for b in range(batch_size):
                        for k in range(K):
                            partial_seqs.append([wids[b][k]])
                else:
                    new_partial_seqs = []
                    for b in range(batch_size):
                        for k in range(K):
                            new_partial_seqs.append(
                                partial_seqs[ptrs[b][k] + b * K] + [wids[b][k]])
                    partial_seqs = new_partial_seqs

                def get_dup_ngram_candidates(seq, n):
                    cands = set()
                    if len(seq) < n:
                        return []
                    tail = seq[-(n - 1):]
                    if self.forbid_ignore_set and any(tk in self.forbid_ignore_set for tk in tail):
                        return []
                    for i in range(len(seq) - (n - 1)):
                        mismatch = False
                        for j in range(n - 1):
                            if tail[j] != seq[i + j]:
                                mismatch = True
                                break
                        if (not mismatch) and not (
                                self.forbid_ignore_set and (seq[i + n - 1] in self.forbid_ignore_set)):
                            cands.add(seq[i + n - 1])
                    return list(sorted(cands))

                if len(partial_seqs[0]) >= self.ngram_size:
                    dup_cands = []
                    for seq in partial_seqs:
                        dup_cands.append(
                            get_dup_ngram_candidates(seq, self.ngram_size))
                    if max(len(x) for x in dup_cands) > 0:
                        if buf_matrix is None:
                            vocab_size = list(log_scores.size())[-1]
                            buf_matrix = np.zeros(
                                (batch_size * K, vocab_size), dtype=float)
                        else:
                            buf_matrix.fill(0)
                        for bk, cands in enumerate(dup_cands):
                            for i, wid in enumerate(cands):
                                buf_matrix[bk, wid] = 1.0
                        forbid_word_mask = torch.tensor(
                            buf_matrix, dtype=log_scores.dtype)
                        forbid_word_mask = torch.reshape(
                            forbid_word_mask, [batch_size * K, 1, vocab_size]).cuda()
                    else:
                        forbid_word_mask = None
            next_pos += 1

        # [(batch, beam)]
        total_scores = [x.tolist() for x in total_scores]
        step_ids = [x.tolist() for x in step_ids]
        step_back_ptrs = [x.tolist() for x in step_back_ptrs]
        # back tracking
        traces: dict = {'pred_seq': [], 'scores': [], 'wids': [], 'ptrs': []}
        for b in range(batch_size):
            # [(beam,)]
            scores = [x[b] for x in total_scores]
            wids_list = [x[b] for x in step_ids]
            ptrs = [x[b] for x in step_back_ptrs]
            traces['scores'].append(scores)
            traces['wids'].append(wids_list)
            traces['ptrs'].append(ptrs)
            # first we need to find the eos frame where all symbols are eos
            # any frames after the eos frame are invalid
            last_frame_id = len(scores) - 1
            for i, wids in enumerate(wids_list):
                if all(wid == self.eos_id for wid in wids):
                    last_frame_id = i
                    break
            max_score = -math.inf
            frame_id = -1
            pos_in_frame = -1

            for fid in range(last_frame_id + 1):
                for i, wid in enumerate(wids_list[fid]):
                    if wid == self.eos_id or fid == last_frame_id:
                        s = scores[fid][i]
                        if self.length_penalty > 0:
                            s /= math.pow((5 + fid + 1) / 6.0,
                                          self.length_penalty)
                        if s > max_score:
                            max_score = s
                            frame_id = fid
                            pos_in_frame = i
            if frame_id == -1:
                traces['pred_seq'].append([0])
            else:
                seq = [wids_list[frame_id][pos_in_frame]]
                for fid in range(frame_id, 0, -1):
                    pos_in_frame = ptrs[fid][pos_in_frame]
                    seq.append(wids_list[fid - 1][pos_in_frame])
                seq.reverse()
                traces['pred_seq'].append(seq)

        def _pad_sequence(sequences, max_len, padding_value=0):
            trailing_dims = sequences[0].size()[1:]
            out_dims = (len(sequences), max_len) + trailing_dims

            out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
            for i, tensor in enumerate(sequences):
                length = tensor.size(0)
                # use index notation to prevent duplicate references to the tensor
                out_tensor[i, :length, ...] = tensor
            return out_tensor

        # convert to tensors for DataParallel
        for k in ('pred_seq', 'scores', 'wids', 'ptrs'):
            ts_list = traces[k]
            if not isinstance(ts_list[0], torch.Tensor):
                dt = torch.float if k == 'scores' else torch.long
                ts_list = [torch.tensor(it, dtype=dt) for it in ts_list]
            traces[k] = _pad_sequence(
                ts_list, output_length, padding_value=0).to(input_ids.device)

        return traces
