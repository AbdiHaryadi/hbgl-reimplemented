import torch

def batch_arange_like(x: torch.Tensor):
    batch_size = x.shape[0]
    length = x.shape[1]

    x = torch.arange(length, device=x.device)
    x = x.repeat(batch_size, 1)
    return x

def prepare_mask_inputs_from_label(
    label_input_ids: torch.Tensor,
    label_attention_mask: torch.Tensor,
    mask_token_id: int,
):
    input_ids = torch.full_like(label_input_ids, mask_token_id)
    attention_mask = label_attention_mask
    return input_ids, attention_mask

def prepare_token_type_ids_from_input_ids(
    text_input_ids: torch.Tensor,
    label_input_ids: torch.Tensor,
    mask_input_ids: torch.Tensor,
):
    text_type_ids = torch.zeros_like(text_input_ids)
    label_type_ids = torch.ones_like(label_input_ids)
    mask_type_ids = torch.ones_like(mask_input_ids)
    return torch.cat([text_type_ids, label_type_ids, mask_type_ids], dim=1)

def prepare_position_ids_from_input_ids(
    text_input_ids: torch.Tensor,
    label_input_ids: torch.Tensor,
    mask_input_ids: torch.Tensor,
):
    text_position_ids = batch_arange_like(text_input_ids)
    label_position_ids = batch_arange_like(label_input_ids) + 1
    mask_position_ids = batch_arange_like(mask_input_ids) + 1

    return torch.cat([text_position_ids, label_position_ids, mask_position_ids], dim=1)

def prepare_4d_attention_mask(
    text_attention_mask_2d: torch.Tensor,
    label_attention_mask_2d: torch.Tensor,
    mask_attention_mask_2d: torch.Tensor,
):
    device = text_attention_mask_2d.device
    assert label_attention_mask_2d.device == device
    assert mask_attention_mask_2d.device == device

    base_attention_mask = torch.cat([text_attention_mask_2d, label_attention_mask_2d, mask_attention_mask_2d], dim=1)
    base_attention_mask = base_attention_mask.unsqueeze(1)

    # per_row_attention_mask = base_attention_mask.unsqueeze(3)
    per_column_attention_mask = base_attention_mask.unsqueeze(2)

    batch_size = base_attention_mask.shape[0]

    text_length = text_attention_mask_2d.shape[1]
    label_length = label_attention_mask_2d.shape[1]
    mask_length = mask_attention_mask_2d.shape[1]
    length = text_length + label_length + mask_length
    
    label_start_index = text_length
    label_stop_index = label_start_index + label_length
    mask_start_index = label_stop_index
    mask_stop_index = mask_start_index + mask_length

    label_to_label_subattention = torch.ones((label_length, label_length), device=device)
    label_to_label_subattention = torch.tril(label_to_label_subattention)

    label_to_mask_subattention = torch.ones((mask_length, label_length))
    label_to_mask_subattention = torch.tril(label_to_mask_subattention, diagonal=-1)

    mask_to_mask_subattention = torch.eye(mask_length)

    text_attention = torch.ones((batch_size, 1, length, text_length), device=device)

    label_attention = torch.zeros((batch_size, 1, length, label_length), device=device)
    label_attention[:, :, label_start_index:label_stop_index, :] = label_to_label_subattention
    label_attention[:, :, mask_start_index:mask_stop_index, :] = label_to_mask_subattention

    mask_attention = torch.zeros((batch_size, 1, length, mask_length), device=device)
    mask_attention[:, :, mask_start_index:mask_stop_index, :] = mask_to_mask_subattention

    bool_attention_mask = torch.cat([text_attention, label_attention, mask_attention], dim=3)
    # bool_attention_mask = bool_attention_mask * per_row_attention_mask * per_column_attention_mask
    bool_attention_mask = bool_attention_mask * per_column_attention_mask
    attention_mask = torch.where(
        bool_attention_mask.bool(),
        torch.tensor(0.0, device=bool_attention_mask.device),
        torch.tensor(-1e9, device=bool_attention_mask.device),
    )
    return attention_mask

def prepare_4d_boolean_attention_mask_with_size(
    batch_size: int,
    text_length: int,
    label_length: int,
    mask_length: int,
):
    length = text_length + label_length + mask_length
    
    label_start_index = text_length
    label_stop_index = label_start_index + label_length
    mask_start_index = label_stop_index
    mask_stop_index = mask_start_index + mask_length

    label_to_label_subattention = torch.ones((label_length, label_length))
    label_to_label_subattention = torch.tril(label_to_label_subattention)

    label_to_mask_subattention = torch.ones((mask_length, label_length))
    label_to_mask_subattention = torch.tril(label_to_mask_subattention, diagonal=-1)

    mask_to_mask_subattention = torch.eye(mask_length)

    text_attention = torch.ones((batch_size, 1, length, text_length))

    label_attention = torch.zeros((batch_size, 1, length, label_length))
    label_attention[:, :, label_start_index:label_stop_index, :] = label_to_label_subattention
    
    label_attention[:, :, mask_start_index:mask_stop_index, :] = label_to_mask_subattention

    mask_attention = torch.zeros((batch_size, 1, length, mask_length))
    mask_attention[:, :, mask_start_index:mask_stop_index, :] = mask_to_mask_subattention

    bool_attention_mask = torch.cat([text_attention, label_attention, mask_attention], dim=3)
    return bool_attention_mask

def batch_arange(batch_size: int, length: int):
    x = torch.arange(length)
    x = x.repeat(batch_size, 1)
    return x

def prepare_token_type_ids_from_size(
    batch_size: int,
    text_length: int,
    label_length: int,
    mask_length: int,
):
    text_type_ids = torch.zeros((batch_size, text_length))
    label_type_ids = torch.ones((batch_size, label_length))
    mask_type_ids = torch.ones((batch_size, mask_length))
    
    result = torch.cat([text_type_ids, label_type_ids, mask_type_ids], dim=1)
    result = result.long()
    return result

def prepare_position_ids_from_size(
    batch_size: int,
    text_length: int,
    label_length: int,
    mask_length: int,
):
    text_position_ids = batch_arange(batch_size, text_length)
    label_position_ids = batch_arange(batch_size, label_length) + 1
    mask_position_ids = batch_arange(batch_size, mask_length) + 1

    result = torch.cat([text_position_ids, label_position_ids, mask_position_ids], dim=1)
    result = result.long()
    return result
