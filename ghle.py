from collections import defaultdict

import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.optim import AdamW
from transformers import BertForMaskedLM


def train(args, tokenizer, input_ids, attention_mask,  position_ids, initial_label_embeddings, num_hiers, reversed_hiers):
    model = load_model_for_train(args)
    label_embeddings = initial_label_embeddings.float().cuda().requires_grad_()
    torch.save(label_embeddings.cpu(), 'before.pt')

    optimizer = init_optimizer(args, label_embeddings)
    for step in range(total_steps(args)):
        current_mask_ratio = calculate_current_mask_ratio(args, step)
        masked_inputs_embeds, mask_tokens = init_masked_input_embeds(args, tokenizer, input_ids, model, label_embeddings, current_mask_ratio)
        prediction_scores = calculate_predictions_scores(args, attention_mask, position_ids, model, label_embeddings, masked_inputs_embeds)
        labels = init_labels(args, input_ids, model, mask_tokens)
        masked_lm_loss = calculate_masked_lm_loss_v2(args, input_ids, num_hiers, reversed_hiers, mask_tokens, prediction_scores, labels)
        optimize_step(model, label_embeddings, optimizer, masked_lm_loss)
        print(f'step {step}', masked_lm_loss.item())

    torch.save(label_embeddings.cpu(), 'after.pt')
    return label_embeddings

def total_steps(args):
    return args.label_cpt_steps

def init_optimizer(args, label_embeddings):
    optimizer_grouped_parameters = [
        {'params': [label_embeddings, ], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.label_cpt_lr, eps=args.adam_epsilon)
    return optimizer

def load_model_for_train(args):
    model = BertForMaskedLM.from_pretrained(args.model_name_or_path)
    model = model.train()
    model.cuda()  # type: ignore
    return model

def optimize_step(model, label_embeddings, optimizer, loss):
    loss.backward()
    optimizer.step()
    model.zero_grad()
    label_embeddings.grad = None

def calculate_masked_lm_loss_v2(args, input_ids, num_hiers, reversed_hiers, mask_tokens, prediction_scores, labels):
    label_nums = input_ids.shape[0] - 2
    if args.label_cpt_use_bce:
        loss_fct = BCEWithLogitsLoss()  # -100 index = padding token
        with torch.no_grad():
            bce_labels = torch.zeros_like(prediction_scores)
            _bce_labels = []
            for b in range(args.label_cpt_bsz):
                l = labels[b][mask_tokens[b]].tolist()
                bce_l = bce_labels[b][mask_tokens[b]]
                c = defaultdict(list)
                lmap = {}
                for il in l:
                    if il not in num_hiers:
                            # last labels
                        p = reversed_hiers[il]
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
        masked_lm_loss = loss_fct(prediction_scores.view(-1, label_nums), labels.view(-1))
    return masked_lm_loss

def init_labels(args, input_ids, model, mask_tokens):
    batched_input_ids = init_batched_input_ids(args, input_ids)
    labels = torch.ones_like(batched_input_ids).long() * -100
    labels[mask_tokens] = batched_input_ids[mask_tokens] - model.bert.embeddings.word_embeddings.num_embeddings
    return labels

def calculate_predictions_scores(args, attention_mask, position_ids, model, label_embeddings, masked_inputs_embeds):
    batched_position_ids = init_batched_position_ids(args, position_ids)
    batched_attention_mask = init_batched_attention_mask(args, attention_mask)
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

def init_masked_input_embeds(args, tokenizer, input_ids, model, label_embeddings, current_mask_ratio):
    batched_input_ids = init_batched_input_ids(args, input_ids)
    mask_tokens = generate_mask_tokens(batched_input_ids, current_mask_ratio)
    masked_inputs_embeds = torch.cat([model.bert.embeddings.word_embeddings.weight[tokenizer.cls_token_id].unsqueeze(0),
                                    label_embeddings,
                                    model.bert.embeddings.word_embeddings.weight[tokenizer.sep_token_id].unsqueeze(0),])
    masked_inputs_embeds = masked_inputs_embeds.unsqueeze(0).repeat(args.label_cpt_bsz, 1, 1).cuda()
    masked_inputs_embeds[mask_tokens] = model.bert.embeddings.word_embeddings.weight[tokenizer.mask_token_id]
    return masked_inputs_embeds, mask_tokens

def generate_mask_tokens(batched_input_ids, current_mask_ratio):
    mask_tokens = ~torch.bernoulli(torch.ones_like(batched_input_ids) * (1 - current_mask_ratio)).bool()
    # keep cls & sep unmask
    mask_tokens[:, 0] = 0
    mask_tokens[:, -1] = 0
    return mask_tokens

def calculate_current_mask_ratio(args, step):
    mask_ratio = 0.15
    if args.label_cpt_not_incr_mask_ratio:
        current_mask_ratio = mask_ratio
    else:
        current_mask_ratio = mask_ratio + (step / args.label_cpt_steps) * 0.3
    return current_mask_ratio

def init_batched_attention_mask(args, attention_mask):
    batch_size = args.label_cpt_bsz
    attention_mask = attention_mask.unsqueeze(0).repeat(batch_size, 1, 1).unsqueeze(1).cuda()
    attention_mask = (1.0 - attention_mask) * -10000.0
    return attention_mask

def init_batched_input_ids(args, input_ids):
    batch_size = args.label_cpt_bsz
    batched_input_ids = input_ids.unsqueeze(0).repeat(batch_size, 1).cuda().long()
    return batched_input_ids

def init_batched_position_ids(args, position_ids):
    batch_size = args.label_cpt_bsz
    position_ids = position_ids.unsqueeze(0).repeat(batch_size, 1).cuda().long()

    if args.label_cpt_decodewithpos:
        position_ids[:, 1:-1] += args.max_source_seq_length - 1
        position_ids[:, -1] = args.max_source_seq_length + args.max_target_seq_length - 1
    return position_ids
