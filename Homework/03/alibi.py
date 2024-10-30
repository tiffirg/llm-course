import math
import torch


def compute_alibi(num_heads: int, seq_len: int) -> torch.Tensor:
    """
    Compute ALiBi for a sequence.

    ALiBi can be used not only with causal models.
    In this case, the biases will be symmetrical about the diagonal up to the sign.

    Args:
        num_heads (int): Number of attention heads.
        seq_len (int): Sequence length.

    Returns:
        torch.Tensor: A tensor containing ALiBi to be added to attention scores.
    """

    # slopes = torch.pow(value, -torch.arange(0, num_heads).float() / num_heads)
    # dist = torch.arange(seq_len).unsqueeze(0) - torch.arange(seq_len).unsqueeze(1)

    alibi = torch.zeros((num_heads, seq_len, seq_len), dtype=torch.float32)
    n = 2 ** math.floor(math.log2(num_heads))
    value = 2.0 ** (8.0 / n)
    if n < num_heads:
        value = 2.0 ** (4.0 / n)
    for k in range(num_heads):
        for i in range(seq_len):
            for j in range(seq_len):
                alibi[k, i, j] = (j - i) / value ** (k + 1)

    return alibi


if __name__ == "__main__":
    bias = compute_alibi(4, 4)
    print(bias)
