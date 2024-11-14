import json
from torch.utils.data import Dataset
import torch
import numpy as np
from torch_geometric.data import Data, Batch
import pickle

class OBQADataset(Dataset):
    def __init__(self, data_path, graph_path, init_emb_path, num_options=4):
        super().__init__()
        self.data = self.load_jsonl(data_path)
        _, self.concept_ids, _, _, _, self.edge_index, self.edge_type, self.half_n_rel, _ = self.load_graph(graph_path)
        self.init_node_emb = torch.FloatTensor(np.load(init_emb_path))
        self.num_options = num_options
        self.concept_ids = list(map(list, zip(*(iter(self.concept_ids),) * self.num_options)))
        self.edge_index = list(map(list, zip(*(iter(self.edge_index),) * self.num_options)))
        self.edge_type = list(map(list, zip(*(iter(self.edge_type),) * self.num_options)))
        self.data_list = self.__preprocess__()

    def __preprocess__(self):
        data_list = []
        for idx in range(len(self.data)):
            concept_ids_list = self.concept_ids[idx]
            edge_index_list, edge_type_list = self.edge_index[idx], self.edge_type[idx]
            graph_list = [
                Data(
                    x=self.init_node_emb[concept_ids_list[i]], 
                    edge_index=edge_index_list[i], 
                    edge_type=edge_type_list[i],
                    concept_ids=concept_ids_list[i],
                ) 
                for i in range(self.num_options)
            ]
            graphs = Batch.from_data_list(graph_list)
            data_list.append(
                {
                    'id':idx,
                    'question': self.data[idx]['question'],
                    'answer': self.data[idx]['answer'],
                    'graphs':graphs,
                }
            )
        return data_list

    def load_jsonl(self, filename):
        data = []
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                data.append(json.loads(line))
        return data

    def load_graph(self, graph_path):
        loaded_data = []
        with open(graph_path, "rb") as in_file:
            try:
                while True:
                    obj = pickle.load(in_file)
                    if type(obj) == dict:
                        assert len(obj) == 1
                        key = list(obj.keys())[0]
                        loaded_data.append(obj[key])
                    elif type(obj) == list:
                        loaded_data.extend(obj)
                    else:
                        raise TypeError("Invalid type for obj.")
            except EOFError:
                pass
        return loaded_data

    def __getitem__(self, idx):
        return self.data_list[idx]

    def __len__(self):
        return len(self.data_list)