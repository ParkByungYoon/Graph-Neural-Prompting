import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.utils import dropout_edge

from src.utils.loss import LinkPrediction


class GraphNeuralPromptModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.gnn = GAT(input_dim, hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim),
        )
        self.cml = CrossModalityLayer(hidden_dim)
        self.link_pred = LinkPrediction(200,38,hidden_dim,hidden_dim)
        self.hidden_dim = hidden_dim
        
    def forward(self, q_emb, graphs):
        device = next(self.gnn.parameters()).device
        x = graphs.x.to(device)
        edge_index = graphs.edge_index.to(device)
        graph_index = graphs.batch.to(device)

        h1 = self.gnn(x, edge_index)
        t = self.ffn(q_emb)
        h3 = self.cml(t,h1)
        h4 = global_mean_pool(h3, graph_index.squeeze())
        return h4
    
    def link_pred_loss(self, graphs, p):
        device = next(self.gnn.parameters()).device
        node_features = graphs.x.squeeze().to(device)
        edge_index = graphs.edge_index.squeeze().to(device)
        edge_type = graphs.edge_type.squeeze().to(device)

        # dropout link
        edge_index, edge_mask = dropout_edge(edge_index, p)
        edge_type = edge_type[edge_mask]

        node_embedding = self.gnn(node_features, edge_index)
        return self.link_pred.loss(edge_index[0], edge_type, edge_index[1], node_embedding)
    
class GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=3):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        for i in range(num_layers):
            conv = GATConv(input_dim, hidden_dim)
            self.convs.append(conv)
            input_dim = hidden_dim

    def forward(self, x, edge_index):
        x = x.view(-1, self.input_dim)
        edge_index = edge_index.view(2,-1)

        for conv in self.convs:
            x = F.relu((conv(x, edge_index)))
        
        return x

class CrossModalityLayer(nn.Module):
    def __init__(self, embedding_dim, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embedding_dim, 1, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embedding_dim, 1, dropout=dropout, batch_first=True)

    def forward(self, text_embedding, node_embedding):
        t = text_embedding
        h1 = node_embedding
        h2,_ = self.self_attn(h1, h1, h1)
        h3,_ = self.cross_attn(h2, t, t)
        return h3

class ProjectionLayer(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.projection_module = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 2048),
        )
        
    def forward(self, x):
        return self.projection_module(x)