import torch.nn as nn
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel


class TextNeuralPromptModel(torch.nn.Module):
    def __init__(self, pretrained_model_name_or_path):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.text_model = AutoModel.from_pretrained(pretrained_model_name_or_path)
        
    def forward(self, graphs):
        encoded_input = self.tokenizer(graphs, padding=True, truncation=True, return_tensors='pt')
        model_output = self.model(**encoded_input)
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        return sentence_embeddings

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

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