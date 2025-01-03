import numpy as np
import torch
from scipy.linalg import expm

def get_adj_matrix(x, edge_index) -> np.ndarray:
    num_nodes = x.shape[0]
    adj_matrix = np.zeros(shape=(num_nodes, num_nodes))
    for i, j in zip(edge_index[0], edge_index[1]):
        adj_matrix[i, j] = 1.
    return adj_matrix

def get_heat_matrix(
        adj_matrix: np.ndarray,
        t: float = 5.0) -> np.ndarray:
    num_nodes = adj_matrix.shape[0]
    A_tilde = adj_matrix + np.eye(num_nodes)
    D_tilde = np.diag(1/np.sqrt(A_tilde.sum(axis=1)))
    H = D_tilde @ A_tilde @ D_tilde
    return expm(-t * (np.eye(num_nodes) - H))

def get_top_k_matrix(A: np.ndarray, k: int = 128) -> np.ndarray:
    num_nodes = A.shape[0]
    row_idx = np.arange(num_nodes)
    A[A.argsort(axis=0)[:num_nodes - k], row_idx] = 0.
    norm = A.sum(axis=0)
    norm[norm <= 0] = 1 # avoid dividing by zero
    return A/norm

def get_ppr_matrix(
        adj_matrix: np.ndarray,
        alpha: float = 0.05) -> np.ndarray:
    num_nodes = adj_matrix.shape[0]
    A_tilde = adj_matrix + np.eye(num_nodes)
    D_tilde = np.diag(1/np.sqrt(A_tilde.sum(axis=1)))
    H = D_tilde @ A_tilde @ D_tilde
    return alpha * np.linalg.inv(np.eye(num_nodes) - (1 - alpha) * H) # numpy, [N, N]

def get_clipped_matrix(A: np.ndarray, eps: float = 0.01) -> np.ndarray:
    num_nodes = A.shape[0]
    A[A < eps] = 0.
    norm = A.sum(axis=0)
    norm[norm <= 0] = 1 # avoid dividing by zero
    return A/norm

def get_PPR_adj(x, edge_index, alpha=0.05, k=None, eps=None):
    assert ((k==None and eps!=None) or (k!=None and eps==None))

    adj_matrix = get_adj_matrix(x, edge_index)
    ppr_matrix = get_ppr_matrix(adj_matrix, alpha=alpha)

    if k!=None:
        ppr_matrix = get_top_k_matrix(ppr_matrix, k=k)
    elif eps!=None:
        ppr_matrix = get_clipped_matrix(ppr_matrix, eps=eps)
    else:
        raise ValueError
    
    return torch.tensor(ppr_matrix).float().to(x.device)


