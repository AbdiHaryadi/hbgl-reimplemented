from dataclasses import dataclass

import torch
import torch.nn as nn
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer

from hbgl.config import Hierarchy

@dataclass
class LabelEmbeddingsConfig:
    lr: float = 1e-3
    training_steps: int = 300
    batch_size: int = 1
    initial_mask_ratio: float = 0.15
    mask_ratio_upper_bound: float = 0.45
    loss_reduction: str = "sum"

    def __post_init__(self):
        for name, value in [
            ("lr", self.lr),
            ("batch_size", self.batch_size),
        ]:
            if not (value > 0):
                raise ValueError(f"Invalid {name}: {value}")
            
        for name, value in [
            ("training_steps", self.training_steps),
        ]:
            if not (value >= 0):
                raise ValueError(f"Invalid {name}: {value}")
            
        for name, value in [
            ("initial_mask_ratio", self.initial_mask_ratio),
            ("mask_ratio_upper_bound", self.mask_ratio_upper_bound),
        ]:
            if not (0 <= value <= 1):
                raise ValueError(f"Invalid {name}: {value}")
            
        if not (self.loss_reduction in ["sum", "mean"]):
            raise ValueError(f"Invalid loss_reduction: {self.loss_reduction}")

def init_label_embeddings_with_averaging(
    token_embedding: nn.Embedding,
    tokenizer: PreTrainedTokenizer,
    label_names: list[str],
):
    device = token_embedding.weight.device

    embeddings: list[torch.Tensor] = []
    for name in label_names:
        ids = tokenizer.encode(name, add_special_tokens=False)
        ids = torch.tensor(ids)
        ids = ids.to(device)

        with torch.no_grad():
            embeds = token_embedding(ids)
        
        average_vector = embeds.mean(dim=0).cpu()
        embeddings.append(average_vector)

    result = torch.stack(embeddings)
    return result

def init_2d_attention_mask(hierarchy: Hierarchy):
    labels = hierarchy.get_labels()
    num_labels = len(labels)
    result = []
    for i in range(num_labels):
        row = []
        for j in range(num_labels):
            if (
                i == j
                or hierarchy.has_child_relation(labels[i], labels[j])
                or hierarchy.has_child_relation(labels[j], labels[i])
            ):
                row.append(0)
            else:
                row.append(1)
        result.append(row)
    return torch.tensor(result)

def prepare_input_masked_label_embeddings(
        bert_embeddings: nn.Module,
        bert_token_embedding: nn.Embedding,
        mask_token_id: int,
        hierarchy: Hierarchy,
        label_embeddings: torch.Tensor,
        masked: torch.Tensor,
):
    # This thing should be executed in batch.
    device = label_embeddings.device
    assert masked.device == device

    with torch.no_grad():
        mask_token_vector = bert_token_embedding(torch.tensor(mask_token_id, device=device))
    position_ids = torch.tensor(hierarchy.get_levels(), device=device)
    token_type_ids = torch.ones_like(position_ids)

    batch_size = masked.shape[0]
    new_label_embeddings = label_embeddings.repeat(batch_size, 1, 1)
    new_label_embeddings[masked] = mask_token_vector.to(device)

    input_embeddings = bert_embeddings(
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        inputs_embeds=new_label_embeddings,
    )
    return input_embeddings

def generate_input_embeddings(
        bert_token_embedding: nn.Embedding,
        bert_embeddings: nn.Module,
        mask_token_id: int,
        hierarchy: Hierarchy,
        config: LabelEmbeddingsConfig,
        label_embeddings: torch.Tensor,
        mask_ratio: float,
):
    num_labels = label_embeddings.shape[0]
    
    prob_values = torch.full((config.batch_size, num_labels), mask_ratio)
    masked = torch.bernoulli(prob_values)
    masked = masked.bool()
    masked = masked.to(label_embeddings.device)

    return prepare_input_masked_label_embeddings(
        bert_embeddings=bert_embeddings,
        bert_token_embedding=bert_token_embedding,
        mask_token_id=mask_token_id,
        hierarchy=hierarchy,
        label_embeddings=label_embeddings,
        masked=masked,
    ), masked

def prepare_targets(hierarchy: Hierarchy, masked: torch.Tensor):
    label_names = hierarchy.get_labels()

    y_bar = []
    for masked_row in masked:
        num_hier_labels = len(label_names)
        for i in range(num_hier_labels):
            if not masked_row[i]:
                continue

            label_i = label_names[i]
            y_hat_row = []
            for j in range(num_hier_labels):
                label_j = label_names[j]
                if (
                        i == j
                        or (
                            masked_row[j]
                            and hierarchy.is_sibling(label_i, label_j)
                            and hierarchy.is_leaf(label_i)
                            and hierarchy.is_leaf(label_j)
                        )
                    ):
                    y_hat_row.append(1.0)
                else:
                    y_hat_row.append(0.0)
                
            y_bar.append(y_hat_row)
    
    return torch.tensor(y_bar)

def init_label_embeddings(
    bert_embeddings: nn.Module,
    bert_encoder: nn.Module,
    tokenizer: PreTrainedTokenizer,
    hierarchy: Hierarchy,
    config: LabelEmbeddingsConfig,
):
    device = next(bert_embeddings.parameters()).device
    for param in bert_embeddings.parameters():
        assert param.device.type == device.type
    
    for param in bert_encoder.parameters():
        assert param.device.type == device.type

    bert_token_embedding = bert_embeddings.word_embeddings
    if not isinstance(bert_token_embedding, nn.Embedding):
        raise ValueError("Cannot extract BERT token embedding")

    label_names = hierarchy.get_labels()
    label_embeddings = init_label_embeddings_with_averaging(
        token_embedding=bert_token_embedding,
        tokenizer=tokenizer,
        label_names=label_names,
    )
    label_embeddings = label_embeddings.to(device)
    label_embeddings_parameter = nn.Parameter(label_embeddings)

    attention_mask = init_2d_attention_mask(hierarchy)
    attention_mask = attention_mask.to(device)
    
    mask_token_id = tokenizer.mask_token_id
    if not isinstance(mask_token_id, int):
        raise ValueError(f"Invalid non-int mask_token_id: {mask_token_id}")

    optimizer = torch.optim.AdamW(
        params=[label_embeddings_parameter],
        lr=config.lr
    )
    criterion = nn.BCEWithLogitsLoss(reduction=config.loss_reduction)

    history: list[float] = []
    pbar = tqdm(range(config.training_steps))
    pbar.set_description(f"Epoch 0 - Loss: N/A")

    for t in pbar:
        alpha = t / config.training_steps
        mask_ratio = (
            (1 - alpha) * config.initial_mask_ratio
            + alpha * config.mask_ratio_upper_bound
        )

        optimizer.zero_grad()
        input_embeddings, masked = generate_input_embeddings(
            bert_token_embedding=bert_token_embedding,
            bert_embeddings=bert_embeddings,
            mask_token_id=mask_token_id,
            hierarchy=hierarchy,
            config=config,
            label_embeddings=label_embeddings,
            mask_ratio=mask_ratio,
        )
        encoder_outputs = bert_encoder(
            hidden_states=input_embeddings,
            attention_mask=attention_mask,
        )
        h: torch.Tensor = encoder_outputs.last_hidden_state
        h = h.to(device)
        logit_s = h @ label_embeddings.T
        masked_logit_s = logit_s[masked]

        y_bar = prepare_targets(hierarchy, masked)
        y_bar = y_bar.to(device)
        loss: torch.Tensor = criterion(masked_logit_s, y_bar)

        loss.backward()
        optimizer.step()

        mask_num = masked.int().sum().cpu().item()
        loss_value = loss.cpu().item()
        if config.loss_reduction == "sum" and mask_num > 0:
            loss_value = loss_value / mask_num
        pbar.set_description(f"Epoch {t + 1} - Loss: {loss_value:.4f}")
        history.append(loss_value)

    return label_embeddings, history
