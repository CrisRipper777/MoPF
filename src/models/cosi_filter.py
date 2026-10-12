from __future__ import annotations

import torch
from torch_geometric.utils import scatter


def sparse_operator_matmul(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """Apply a COO operator whose entries use the source-to-target convention."""
    if edge_index.numel() == 0:
        return torch.zeros_like(x)
    src, dst = edge_index
    messages = x[src] * edge_weight.to(dtype=x.dtype).unsqueeze(-1)
    return scatter(messages, dst, dim=0, dim_size=x.size(0), reduce="sum")


def chebyshev_bank(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    max_order: int = 3,
) -> torch.Tensor:
    """Build T_k(P)X by the recurrence for a normalized adjacency P.

    This is the standard Chebyshev polynomial recurrence applied directly to
    P. It is intentionally distinct from ChebConv's scaled-Laplacian setup.
    The returned shape is ``[N, max_order + 1, D]``.
    """
    if max_order < 0:
        raise ValueError(f"max_order must be >= 0, got {max_order}")
    states = [x]
    if max_order == 0:
        return torch.stack(states, dim=1)
    previous = x
    current = sparse_operator_matmul(edge_index, edge_weight, x)
    states.append(current)
    for _ in range(2, max_order + 1):
        following = 2.0 * sparse_operator_matmul(edge_index, edge_weight, current) - previous
        states.append(following)
        previous, current = current, following
    return torch.stack(states, dim=1)


__all__ = ["chebyshev_bank", "sparse_operator_matmul"]
