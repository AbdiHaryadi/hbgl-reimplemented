from collections import defaultdict
from dataclasses import dataclass

import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.optim import AdamW
from transformers import BertForMaskedLM, BertTokenizer

from args import HBGLTrainingArguments
from hierarchy import LabelHierarchy, LabelMapper

def prepare_attention_mask(
        hierarchy: LabelHierarchy,
        label_mapper: LabelMapper,
        label_cpt_use_bce=False
):
    # cls l1 l2 l3 sep
    attention_mask = torch.zeros((len(label_mapper) + 2, len(label_mapper) + 2))
    for parent_name, child_name in hierarchy.iter_parent_child_name_pairs():
        parent_id = label_mapper.label_to_id(parent_name)
        child_id = label_mapper.label_to_id(child_name)
        # Here we use +1 because CLS is in index zero, and the first ID in hierarchy label is also zero.
        attention_mask[parent_id + 1][child_id + 1] = 1
        if label_cpt_use_bce:
            attention_mask[child_id + 1][parent_id + 1] = 1

    return attention_mask

def get_num_hiers(hierarchy: LabelHierarchy, label_mapper: LabelMapper) -> dict[int, list[int]]:
    num_hiers = defaultdict(list)
    for parent, child in hierarchy.iter_parent_child_name_pairs():
        num_hiers[label_mapper.label_to_id(parent) + 1].append(label_mapper.label_to_id(child) + 1)
    return num_hiers

def get_reversed_num_hiers(hierarchy: LabelHierarchy, label_mapper: LabelMapper) -> dict[int, int]:
    reversed_num_hiers: dict[int, int] = {}
    for parent, child in hierarchy.iter_parent_child_name_pairs():
        reversed_num_hiers[label_mapper.label_to_id(child) + 1] = label_mapper.label_to_id(parent) + 1
    return reversed_num_hiers

@dataclass
class GHLETrainingArguments:
    model_name_or_path: str
    label_mapper_path: str
    hierarchy_path: str
    
    max_source_seq_length: int = 464
    max_target_seq_length: int = 48
    adam_epsilon: float = 1e-8

    learning_rate: float = 1e-3
    max_steps: int = 500
    batch_size: int = 32
    increased_mask_ratio: bool = True
    use_bce: bool = False

    @classmethod
    def from_args(cls, args: HBGLTrainingArguments):
        assert args.add_vocab_file is not None
        assert args.label_cpt is not None
        return cls(
            model_name_or_path=args.model_name_or_path,
            label_mapper_path=args.add_vocab_file,
            hierarchy_path=args.label_cpt,
            max_source_seq_length=args.max_source_seq_length,
            max_target_seq_length=args.max_target_seq_length,
            adam_epsilon=args.adam_epsilon,
            learning_rate=args.label_cpt_lr,
            max_steps=args.label_cpt_steps,
            batch_size=args.label_cpt_bsz,
            increased_mask_ratio=not args.label_cpt_not_incr_mask_ratio,
            use_bce=args.label_cpt_use_bce,
        )

class GHLETrainer:
    def __init__(
            self,
            args: GHLETrainingArguments,
            tokenizer: BertTokenizer,
    ):
        self.args = args
        self.tokenizer = tokenizer

        hierarchy = LabelHierarchy.load(args.hierarchy_path)
        label_mapper = LabelMapper.load(args.label_mapper_path)

        self.num_hiers = get_num_hiers(hierarchy, label_mapper)
        self.reversed_num_hiers = get_reversed_num_hiers(hierarchy, label_mapper)

        self.attention_mask = prepare_attention_mask(hierarchy, label_mapper, args.use_bce)

        self.input_ids = torch.tensor(tokenizer.encode(' '.join(label_mapper.get_all_tokens()).lower()), dtype=torch.long)
        assert len(self.input_ids) == len(label_mapper) + 2

        label_depth = {label: len(hierarchy.find_path(label)) for label in hierarchy.iter_unique_child_names()}
        self.position_ids = torch.tensor([0, ] + [label_depth[label] for label in label_mapper.get_all_labels()] + [max(label_depth.values()) + 1,], dtype=torch.long)

        # Actually, we don't get the device from GHLETrainingArguments, so we duplicate the method from args.py
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, initial_label_embeddings: torch.Tensor):
        model = self.load_model_for_train()
        label_embeddings = initial_label_embeddings.float().to(self.device).requires_grad_()
        torch.save(label_embeddings.cpu(), 'before.pt')
    
        optimizer = self.init_optimizer(label_embeddings)
        for step in range(self.args.max_steps):
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
        model.train()
        model.to(self.device)  # type: ignore
        return model

    def init_optimizer(self, label_embeddings):
        return AdamW(
            params=[{'params': [label_embeddings], 'weight_decay': 0.0}],
            lr=self.args.learning_rate,
            eps=self.args.adam_epsilon
        )

    def calculate_current_mask_ratio(self, step):
        mask_ratio = 0.15
        if self.args.increased_mask_ratio:
            return mask_ratio + (step / self.args.max_steps) * 0.3
        else:
            return mask_ratio

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

    def get_batched_input_ids(self):
        return self.input_ids.unsqueeze(0).repeat(self.args.batch_size, 1).to(self.device).long()

    def get_batched_attention_mask(self):
        batched_attention_mask = self.attention_mask.unsqueeze(0).repeat(self.args.batch_size, 1, 1).unsqueeze(1).to(self.device)
        batched_attention_mask = (1.0 - batched_attention_mask) * -10000.0
        return batched_attention_mask

    def get_batched_position_ids(self):
        return self.position_ids.unsqueeze(0).repeat(self.args.batch_size, 1).to(self.device).long()

    def generate_masked_input_embeds(self, model, label_embeddings, current_mask_ratio):
        mask_tokens = self.generate_mask_tokens(current_mask_ratio)
        masked_inputs_embeds = torch.cat([model.bert.embeddings.word_embeddings.weight[self.tokenizer.cls_token_id].unsqueeze(0),
                                        label_embeddings,
                                        model.bert.embeddings.word_embeddings.weight[self.tokenizer.sep_token_id].unsqueeze(0),])
        masked_inputs_embeds = masked_inputs_embeds.unsqueeze(0).repeat(self.args.batch_size, 1, 1).to(self.device)
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
        if self.args.use_bce:
            loss_fct = BCEWithLogitsLoss()  # -100 index = padding token
            with torch.no_grad():
                bce_labels = torch.zeros_like(prediction_scores)
                _bce_labels = []
                for b in range(self.args.batch_size):
                    l = labels[b][mask_tokens[b]].tolist()
                    bce_l = bce_labels[b][mask_tokens[b]]
                    c = defaultdict(list)
                    lmap = {}
                    for il in l:
                        if il not in self.num_hiers:
                                # last labels
                            p = self.reversed_num_hiers[il]
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
