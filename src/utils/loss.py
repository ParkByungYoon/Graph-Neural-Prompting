from torch_geometric.nn.kge import DistMult
from torch import Tensor

import torch.nn.functional as F
import torch.nn as nn
import torch


class LinkPrediction(DistMult):
    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        input_dims: int,
        hidden_channels: int,
        margin: float = 1.0,
        sparse: bool = False,
    ):
        super().__init__(num_nodes, num_relations, hidden_channels, margin, sparse)
        self.linear = nn.Linear(input_dims, hidden_channels)
    
    def forward(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
        node_embedding: Tensor,
    ) -> Tensor:
    
        head = self.linear(node_embedding[head_index])
        rel = self.rel_emb(rel_type)
        tail = self.linear(node_embedding[tail_index])

        return (head * rel * tail).sum(dim=-1)

    def loss(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
        node_embedding: Tensor,
    ) -> Tensor:

        pos_score = self(head_index, rel_type, tail_index, node_embedding)
        neg_score = self(*self.random_sample(head_index, rel_type, tail_index), node_embedding)

        return F.margin_ranking_loss(
            pos_score,
            neg_score,
            target=torch.ones_like(pos_score),
            margin=self.margin,
        )