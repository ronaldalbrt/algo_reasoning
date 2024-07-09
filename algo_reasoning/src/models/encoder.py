import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from algo_reasoning.src.data import CLRSData
from algo_reasoning.src.specs import Stage, Location, Type, SPECS, CATEGORIES_DIMENSIONS

_Tensor = torch.Tensor

def preprocess(data:_Tensor, _type:str, nb_nodes) -> _Tensor:
    if _type != Type.CATEGORICAL:
        if _type == Type.POINTER:
            data = F.one_hot(data.long(), nb_nodes).unsqueeze(-1).to(torch.float32)
        else:
            data = data.unsqueeze(-1)
  
    return data

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
    def __init__(self, algorithm, encode_hints=True, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.algorithm = algorithm
        self.encode_hints = encode_hints
        self.encoder = nn.ModuleDict()

        self.specs = SPECS[algorithm]
        for k, v in self.specs.items():
            stage, loc, type_ = v

            if stage == Stage.OUTPUT:
                continue

            logger.debug(f"Building Encoder for {k}.")

            input_dim = 1
            if type_ == Type.CATEGORICAL:
                input_dim = CATEGORIES_DIMENSIONS[algorithm][k]

            self.encoder[k] = LinearEncoder(input_dim, hidden_dim)

            # Edge Pointers admit encoders on receiver and sender nodes
            if loc == Location.EDGE and type_ == Type.POINTER:
                self.encoder[k+"_2"] = LinearEncoder(input_dim, hidden_dim)

    def _encode_CLRSData(self, data, node_hidden, edge_hidden, graph_hidden, adj_mat, nb_nodes, hint_step=None):    
        for k, value in data:
            if k not in SPECS[self.algorithm]:
                continue

            if hint_step is not None:
                value = value[:, hint_step]
            
            _, loc, type_ = SPECS[self.algorithm][k]

            data = preprocess(value, type_, nb_nodes)

            encoding = self.encoder[k](data)

            if (loc == Location.NODE and type_ != Type.POINTER) or (loc == Location.GRAPH and type_ == Type.POINTER):
                node_hidden += encoding

            elif loc == Location.EDGE or (loc == Location.NODE and type_ == Type.POINTER):
                if loc == Location.EDGE and type_ == Type.POINTER:
                    encoding2 = self.encoder[k+"_2"](data)

                    edge_hidden += torch.mean(encoding, dim=1) + torch.mean(encoding2, dim=2)
                else:
                    edge_hidden += encoding

            elif loc == Location.GRAPH and type_ != Type.POINTER:
                graph_hidden += encoding

            if loc == Location.NODE and type_ == Type.POINTER:
                data = data.squeeze(-1)
                adj_mat += ((data + data.permute((0, 2, 1))) > 0.5)
                
            elif loc == Location.EDGE and type_ == Type.MASK:
                data = data.squeeze(-1)
                adj_mat += ((data + data.permute((0, 2, 1))) > 0.0)

        return node_hidden, edge_hidden, graph_hidden, (adj_mat > 0.).to(torch.float)

    def forward(self, batch, hints=None, hint_step=None):
        batch_size = len(batch.inputs.batch)
        nb_nodes = batch.inputs.pos.shape[1]
        device = batch.inputs.pos.device
        adj_mat = (torch.eye(nb_nodes, device=device)[None, :, :]).repeat(batch_size, 1, 1)
        node_hidden = torch.zeros((batch_size, nb_nodes, self.hidden_dim), device=device)
        edge_hidden = torch.zeros((batch_size, nb_nodes, nb_nodes, self.hidden_dim), device=device)
        graph_hidden = torch.zeros((batch_size, self.hidden_dim), device=device)

        node_hidden, edge_hidden, graph_hidden, adj_mat = self._encode_CLRSData(batch.inputs, node_hidden, edge_hidden, graph_hidden, adj_mat, nb_nodes)

        if self.encode_hints and hints is not None:
            node_hidden, edge_hidden, graph_hidden, adj_mat = self._encode_CLRSData(hints, node_hidden, edge_hidden, graph_hidden, adj_mat, nb_nodes, 
                                                                                    hint_step=hint_step)

        return node_hidden, edge_hidden, graph_hidden, adj_mat
    

    