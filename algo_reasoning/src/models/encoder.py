import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from algo_reasoning.src.data.data import CLRSData
from algo_reasoning.src.data.specs import Stage, Location, Type, SPECS, CATEGORIES_DIMENSIONS

_Tensor = torch.Tensor

def preprocess(data:_Tensor, _type:str, nb_nodes:int=16) -> _Tensor:
    if _type != Type.CATEGORICAL:
        if _type == Type.POINTER:
            data = F.one_hot(data.to(torch.int64), nb_nodes).to(torch.float32)
        else:
            data = data.unsqueeze(-1)
  
    return data

def encode_CLRSData(data:CLRSData, models:nn.ModuleDict, algorithm:str, nb_nodes:int=16,
                    node_hidden=None, edge_hidden=None, graph_hidden=None, hint_step=None) -> _Tensor:    
    for key, value in data:
        if key not in SPECS[algorithm]:
            logger.warning(f"Key {key} not in specs for algorithm {algorithm}")
            continue
        if hint_step is not None:
            value = value[:, hint_step]
        
        _, loc, type_ = SPECS[algorithm][key]

        logger.debug(f"Encoding {key}.")

        encoding = models[key](preprocess(value, type_, nb_nodes))

        print(f"Key: {key} | Encoding: {encoding.shape}")

        if loc == Location.NODE:
            if node_hidden is None:
                node_hidden = encoding
            else:
                node_hidden += encoding

        elif loc == Location.EDGE:
            if edge_hidden is None:
                edge_hidden = encoding
            else:
                edge_hidden += encoding

        elif loc == Location.GRAPH:
            if graph_hidden is None:
                graph_hidden = encoding
            else:
                graph_hidden += encoding

    return node_hidden, edge_hidden, graph_hidden

class LinearEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.lin = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        x = self.lin(x)
        return x


class Encoder(nn.Module):
    def __init__(self, algorithm, encode_hints=True, nb_nodes=16, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.nb_nodes = nb_nodes
        self.algorithm = algorithm
        self.encode_hints = encode_hints
        self.encoder = nn.ModuleDict()

        self.specs = SPECS[algorithm]
        for k, v in self.specs.items():
            stage, loc, type_ = v

            if stage == Stage.OUTPUT:
                continue

            input_dim = 1
            if type_ == Type.CATEGORICAL:
                input_dim = CATEGORIES_DIMENSIONS[algorithm][k]
            elif type_ == Type.POINTER:
                input_dim = nb_nodes        

            self.encoder[k] = LinearEncoder(input_dim, hidden_dim)

    def forward(self, batch, hint_step=None):
        node_hidden = None
        edge_hidden = None
        graph_hidden = None

        node_hidden, edge_hidden, graph_hidden = encode_CLRSData(batch.inputs, self.encoder, self.algorithm, self.nb_nodes)

        if self.encode_hints and hint_step is not None:
            node_hidden, edge_hidden, graph_hidden = encode_CLRSData(batch.hints, self.encoder, self.algorithm, self.nb_nodes,
                                                                     node_hidden=node_hidden, edge_hidden=edge_hidden, graph_hidden=graph_hidden, hint_step=hint_step)

        return node_hidden, edge_hidden, graph_hidden
