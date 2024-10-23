import numpy as np
import torch
import torch.nn.functional as F


def compute_attention(queries, keys, values) -> torch.Tensor:
    """
    queries- (BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM)
    keys- (BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM)
    values- (BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM)
    """
    hidden_dim = queries.shape[2]
    attention_values = F.softmax(torch.matmul(queries, keys.mT) / np.sqrt(hidden_dim), dim=2)
    return torch.matmul(attention_values, values)


def compute_multihead_attention(queries, keys, values, projection_matrix) -> torch.Tensor:
    """
    queries- (BATCH_SIZE, N_HEADS, SEQ_LENGTH, DIM_PER_HEAD)
    keys- (BATCH_SIZE, N_HEADS, SEQ_LENGTH, DIM_PER_HEAD)
    values- (BATCH_SIZE, N_HEADS, SEQ_LENGTH, DIM_PER_HEAD)
    projection_matrix- (N_HEADS*DIM_PER_HEAD, N_HEADS*DIM_PER_HEAD)
    """
    n_heads = queries.shape[1]
    attentions = [compute_attention(queries[:, i, :, :], keys[:, i, :, :], values[:, i, :, :]) for i in range(n_heads)]
    return torch.matmul(torch.concat(attentions, dim=-1), projection_matrix.T)


def compute_rotary_embeddings(x)-> torch.Tensor:
    """
    x- (BATCH_SIZE, SEQ_LENGTH, N_HEADS, DIM_PER_HEAD)
    """
    
    def create_rotation_matrices(values, index):
        matrices = []
        for value in values:
            cos_val, sin_val = np.cos([value * index])[0], np.sin([value * index])[0]
            rotation = torch.tensor([[cos_val, -sin_val], [sin_val, cos_val]])
            matrices.append(rotation)
        return matrices
    
    batch_size, seq_len, n_heads, dim_per_head = x.shape
    values = torch.tensor([10 ** (-8 * idx / dim_per_head) for idx in range(dim_per_head // 2)])
    rotation_matrices = [create_rotation_matrices(values, i) for i in range(seq_len)]
    
    stacked_matrices = torch.stack([torch.block_diag(*rotation_matrices[i]) for i in range(seq_len)])

    reshaped_x = x.permute(0, 2, 1, 3).reshape(batch_size * n_heads, seq_len, dim_per_head)
    transformed_x = torch.zeros_like(reshaped_x)
    
    for i in range(seq_len):
        transformed_x[:, i, :] = torch.matmul(reshaped_x[:, i, :], stacked_matrices[i].T)
    
    result = transformed_x.reshape(batch_size, n_heads, seq_len, dim_per_head).permute(0, 2, 1, 3)
    return result
