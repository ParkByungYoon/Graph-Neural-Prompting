import json
from torch.utils.data import Dataset
import pandas as pd
import os
from tqdm import tqdm

class SceneGraphDataset(Dataset):
    def __init__(self, data_dir):
        super().__init__()
        self.path = os.path.join(data_dir, "sceneGraphs")
        self.path_nodes = f'{self.path}/nodes'
        self.path_edges = f'{self.path}/edges'
        self.data = pd.read_csv(os.path.join(data_dir, "questions.csv"))
        self.scene_graphs = self.load_json(os.path.join(self.path, "scene_graphs_name.json"))
        self.data_list = self.__preprocess__()

    def __preprocess__(self):
        data_list = []
        for idx in tqdm(range(len(self.data))):
            data_list.append(
                {
                    'id':idx,
                    'question': self.data.loc[idx]['question'],
                    'answer': self.data.loc[idx]['answer'],
                    'graph': self.scene_graphs[str(self.data.loc[idx]['image_id'])],
                }
            )
        return data_list

    def load_json(self, file_path):
        with open(file_path, "r") as f:
            scene_graphs = json.load(f)
        return scene_graphs

    def get_idx_split(self):
        with open(f'{self.path}/split/train_indices.txt', 'r') as file:
            train_indices = [int(line.strip()) for line in file]
        with open(f'{self.path}/split/val_indices.txt', 'r') as file:
            val_indices = [int(line.strip()) for line in file]
        with open(f'{self.path}/split/test_indices.txt', 'r') as file:
            test_indices = [int(line.strip()) for line in file]
        return {'train': train_indices, 'val': val_indices, 'test': test_indices}
    
    def __getitem__(self, idx):
        return self.data_list[idx]

    def __len__(self):
        return len(self.data_list)