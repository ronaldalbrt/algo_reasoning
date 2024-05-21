import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from algo_reasoning.src.data.data import CLRSData
from algo_reasoning.src.data.specs import Stage, Location, Type

_Tensor = torch.Tensor

_CATEGORIES_DIMENSIONS = {
    "heapsort": {
        "phase": 3
    },
    "lcs_length": {
        "key": 4,
        'b': 4,
        "b_h": 4
    },
    "dfs": {
        "color": 3
    },
    "topological_sort": {
        "color": 3
    },
    "strongly_connected_components": {
        "color": 3
    },
    "articulation_points": {
        "color": 3
    },
    "bridges": {
        "color": 3
    },
    "mst_kruskal" : {
        "phase": 3
    },
    "dag_shortest_paths": {
        "color": 3
    },
    "naive_string_matcher": {
        "key": 4
    },
    "kmp_matcher": {
        "key": 4
    },
    "graham_scan": {
        "phase": 5
    },
    "jarvis_march": {
        "phase": 2
    }
}

def preprocess(data:_Tensor, _type:str, nb_nodes:int=16) -> _Tensor:
    if _type == Type.POINTER:
        data = F.one_hot(data, nb_nodes)
  
    return data

class LinearEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.lin = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        x = self.lin(x)
        return x


class Encoder(nn.Module):
    def __init__(self, specs, hidden_dim=128):
        super().__init__()
        self.specs = specs
        self.hidden_dim = hidden_dim
        self.encoder = nn.ModuleDict()
        for k, v in specs.items():
            stage, loc, type_ = v

            if loc == Location.NODE and stage == Stage.INPUT:
                input_dim = 1                

            self.encoder[k] = LinearEncoder(input_dim, hidden_dim)
    def forward(self, batch):
        hidden = None
        for key, value in batch.inputs:
            if key not in self.encoder:
                logger.debug(f"Ignoring {key}")
                continue
            logger.debug(f"Encoding {key}")
            encoding = self.encoder[key](value)
            # check of nan
            if torch.isnan(encoding).any():
                logger.warning(f"NaN in encoded hidden state")
                raise Exception(f"NaN in encoded hidden state")
            if hidden is None:
                hidden = encoding
            else:
                hidden += encoding

        return hidden
