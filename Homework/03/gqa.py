import math
import torch


def scaled_dot_product_gqa(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, is_causal: bool = True, need_weights: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot-Product attention in grouped manner.

    Args:
        query (torch.Tensor): Query tensor of shape [batch size; seq len; num heads; hidden dim]
        key (torch.Tensor): Key tensor of shape [batch size; kv seq len; num kv heads; hidden dim]
        value (torch.Tensor): Value tensor of shape [batch size; kv seq len; num kv heads; hidden dim]
        is_causal (bool): Whether causal mask of attention should be used
        need_weights (bool): Whether attention weights should be returned

    Returns:
        2-tuple of torch.Tensor:
            - Attention output with shape [batch size; seq len; num heads; hidden dim]
            - (Optional) Attention weights with shape [batch size; num heads; seq len; kv seq len].
                Only returned if 'need_weights' is True.
    """

    # seq_len, num_heads, hidden_dim = query.shape[1:]
    # kv_seq_len, num_kv_heads = key.shape[1:3]
    # key = key.repeat_interleave(num_heads // num_kv_heads, -3)
    # value = value.repeat_interleave(num_heads // num_kv_heads, -3)
    # scale_factor = 1 / math.sqrt(hidden_dim)

    # attentions_bias = torch.zeros(seq_len, kv_seq_len, dtype=query.dtype)
    # if is_causal:
    #     temp_mask = torch.ones(seq_len, kv_seq_len, dtype=torch.bool).tril(diagonal=0)
    #     attentions_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
    #     attentions_bias.to(query.dtype)

    # attentions = query @ key.transpose(-2, -1) * scale_factor + attentions_bias
    # attentions = torch.softmax(attentions, dim=-1)
    # result = attentions @ value

    seq_len, num_heads, hidden_dim = query.shape[1:]
    kv_seq_len, num_kv_heads = key.shape[1:-1]

    if num_heads % num_kv_heads:
        raise ValueError('Error params')
    query = query.permute(0, 2, 1, 3)
    key = key.repeat_interleave(repeats=num_heads // num_kv_heads, dim=2).permute(0, 2, 1, 3)
    scale_factor = 1 / math.sqrt(hidden_dim)
    attns = query @ key.mT * scale_factor
    if is_causal:
        temp_mask = torch.triu(torch.ones(seq_len, kv_seq_len), diagonal=1).bool()
        attns.masked_fill_(temp_mask, float("-inf"))
        attns.to(query.dtype)
    attns = torch.softmax(attns, dim=-1)
    value = value.repeat_interleave(repeats=num_heads // num_kv_heads, dim=2).permute(0, 2, 1, 3)
    result = (attns @ value).permute(0, 2, 1, 3)
    if need_weights:
        return result, attns
    else:
        return result

