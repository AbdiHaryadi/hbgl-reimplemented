from collections import defaultdict

import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.optim import AdamW
from transformers import BertForMaskedLM


class Trainer:
    def __init__(self, args, tokenizer, input_ids, attention_mask,  position_ids, num_hiers, reversed_hiers):
        self.args = args
        self.tokenizer = tokenizer
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.position_ids = position_ids
        self.num_hiers = num_hiers
        self.reversed_hiers = reversed_hiers

    def train(self, initial_label_embeddings):
        model = self.load_model_for_train()
        label_embeddings = initial_label_embeddings.float().cuda().requires_grad_()
        torch.save(label_embeddings.cpu(), 'before.pt')
    
        optimizer = self.init_optimizer(label_embeddings)
        for step in range(self.get_total_steps()):
            current_mask_ratio = self.calculate_current_mask_ratio(step)
            masked_inputs_embeds, mask_tokens = self.generate_masked_input_embeds(model, label_embeddings, current_mask_ratio)
            prediction_scores = self.calculate_predictions_scores(model, label_embeddings, masked_inputs_embeds)
            labels = self.init_labels(model, mask_tokens)
            masked_lm_loss = self.calculate_masked_lm_loss(mask_tokens, prediction_scores, labels)
            self.optimize_step(model, label_embeddings, optimizer, masked_lm_loss)
            print(f'step {step}', masked_lm_loss.item())
    
        torch.save(label_embeddings.cpu(), 'after.pt')
        return label_embeddings

    def load_model_for_train(self):
        model = BertForMaskedLM.from_pretrained(self.args.model_name_or_path)
        model = model.train()
        model.cuda()  # type: ignore
        return model

    def init_optimizer(self, label_embeddings):
        return AdamW(
            params=[{'params': [label_embeddings], 'weight_decay': 0.0}],
            lr=self.args.label_cpt_lr,
            eps=self.args.adam_epsilon
        )

    def get_total_steps(self):
        return self.args.label_cpt_steps

    def calculate_current_mask_ratio(self, step):
        mask_ratio = 0.15
        if self.args.label_cpt_not_incr_mask_ratio:
            return mask_ratio
        else:
            return mask_ratio + (step / self.args.label_cpt_steps) * 0.3

    def calculate_predictions_scores(self, model, label_embeddings, masked_inputs_embeds):
        batched_position_ids = self.get_batched_position_ids()
        batched_attention_mask = self.get_batched_attention_mask()
        outputs = model.bert(
            None,
            attention_mask=batched_attention_mask,
            position_ids=batched_position_ids,
            inputs_embeds=masked_inputs_embeds,
        )
        sequence_output = outputs[0]
        hidden_states = model.cls.predictions.transform(sequence_output)
        prediction_scores = hidden_states @ label_embeddings.T
        return prediction_scores

    def get_batch_size(self):
        return self.args.label_cpt_bsz

    def get_batched_input_ids(self):
        return self.input_ids.unsqueeze(0).repeat(self.get_batch_size(), 1).cuda().long()

    def get_batched_attention_mask(self):
        batched_attention_mask = self.attention_mask.unsqueeze(0).repeat(self.get_batch_size(), 1, 1).unsqueeze(1).cuda()
        batched_attention_mask = (1.0 - batched_attention_mask) * -10000.0
        return batched_attention_mask

    def get_batched_position_ids(self):
        batched_position_ids = self.position_ids.unsqueeze(0).repeat(self.get_batch_size(), 1).cuda().long()

        if self.args.label_cpt_decodewithpos:
            batched_position_ids[:, 1:-1] += self.args.max_source_seq_length - 1
            batched_position_ids[:, -1] = self.args.max_source_seq_length + self.args.max_target_seq_length - 1
        return batched_position_ids

    def generate_masked_input_embeds(self, model, label_embeddings, current_mask_ratio):
        mask_tokens = self.generate_mask_tokens(current_mask_ratio)
        masked_inputs_embeds = torch.cat([model.bert.embeddings.word_embeddings.weight[self.tokenizer.cls_token_id].unsqueeze(0),
                                        label_embeddings,
                                        model.bert.embeddings.word_embeddings.weight[self.tokenizer.sep_token_id].unsqueeze(0),])
        masked_inputs_embeds = masked_inputs_embeds.unsqueeze(0).repeat(self.get_batch_size(), 1, 1).cuda()
        masked_inputs_embeds[mask_tokens] = model.bert.embeddings.word_embeddings.weight[self.tokenizer.mask_token_id]
        return masked_inputs_embeds, mask_tokens

    def generate_mask_tokens(self, current_mask_ratio):
        mask_tokens = ~torch.bernoulli(
            torch.ones_like(self.get_batched_input_ids())
            * (1 - current_mask_ratio)
        ).bool()
        # keep cls & sep unmask
        mask_tokens[:, 0] = 0
        mask_tokens[:, -1] = 0
        return mask_tokens

    def init_labels(self, model, mask_tokens):
        batched_input_ids = self.get_batched_input_ids()
        labels = torch.ones_like(batched_input_ids).long() * -100
        labels[mask_tokens] = batched_input_ids[mask_tokens] - model.bert.embeddings.word_embeddings.num_embeddings
        return labels

    def calculate_masked_lm_loss(self, mask_tokens, prediction_scores, labels):
        if self.args.label_cpt_use_bce:
            loss_fct = BCEWithLogitsLoss()  # -100 index = padding token
            with torch.no_grad():
                bce_labels = torch.zeros_like(prediction_scores)
                _bce_labels = []
                for b in range(self.args.label_cpt_bsz):
                    l = labels[b][mask_tokens[b]].tolist()
                    bce_l = bce_labels[b][mask_tokens[b]]
                    c = defaultdict(list)
                    lmap = {}
                    for il in l:
                        if il not in self.num_hiers:
                                # last labels
                            p = self.reversed_hiers[il]
                            c[p].append(il)
                            lmap[il] = p
                    for i, il in enumerate(l):
                        if il not in lmap:
                            bce_l[i][il] = 1
                        else:
                            for j in c[lmap[il]]:
                                bce_l[i][j] = 1
                    _bce_labels.append(bce_l)
                bce_labels = torch.cat(_bce_labels, dim=0)
                print(bce_labels.sum())
            masked_lm_loss = loss_fct(prediction_scores[mask_tokens], bce_labels)
        else:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.get_label_nums()), labels.view(-1))
        return masked_lm_loss

    def get_label_nums(self):
        return self.input_ids.shape[0] - 2

    def optimize_step(self, model, label_embeddings, optimizer, loss):
        loss.backward()
        optimizer.step()
        model.zero_grad()
        label_embeddings.grad = None

def train(args, tokenizer, input_ids, attention_mask,  position_ids, initial_label_embeddings, num_hiers, reversed_hiers):
    return Trainer(
        args,
        tokenizer,
        input_ids,
        attention_mask,
        position_ids,
        num_hiers,
        reversed_hiers
    ).train(initial_label_embeddings)
