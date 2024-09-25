import torch
import torch.nn as nn
import torch.nn.functional as F
from algo_reasoning.src.data import CLRSData
from algo_reasoning.src.specs import Stage, Location, Type, SPECS, CATEGORIES_DIMENSIONS

_Tensor = torch.Tensor

def preprocess(data:_Tensor, type_:str, nb_nodes) -> _Tensor:
    if type_ != Type.CATEGORICAL:
        if type_ == Type.POINTER:
            data = F.one_hot(data.long(), nb_nodes).unsqueeze(-1).to(torch.float32)
        else:
            data = data.unsqueeze(-1)
  
    return data

class Encoder(nn.Module):
    def __init__(self, algorithm, encode_hints=True, soft_hints=True, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.algorithm = algorithm
        self.encode_hints = encode_hints
        self.soft_hint = soft_hints
        self.encoder = nn.ModuleDict()

        self.specs = SPECS[algorithm]
        for k, v in self.specs.items():
            stage, loc, type_ = v

            if stage == Stage.OUTPUT:
                continue

            input_dim = 1
            if type_ == Type.CATEGORICAL:
                input_dim = CATEGORIES_DIMENSIONS[algorithm][k]

            self.encoder[k] = nn.Linear(input_dim, hidden_dim)

            # Edge Pointers admit encoders on receiver and sender nodes
            if loc == Location.EDGE and type_ == Type.POINTER:
                self.encoder[k+"_2"] = nn.Linear(input_dim, hidden_dim)

        self.reset_parameters()

    def reset_parameters(self):
        for k, v in self.encoder.items(): 
            stage, _, type_ = self.specs[k]
            m = self.encoder[k]

            if stage == Stage.HINT and type_ == Type.SCALAR:
                nn.init.xavier_normal_(m.weight) 

    def _encode_CLRSData(self, data, node_hidden, edge_hidden, graph_hidden, adj_mat, nb_nodes, hint_step=None):    
        for k, value in data:
            if k not in SPECS[self.algorithm]:
                continue

            if hint_step is not None:
                value = value[:, hint_step]
            
            _, loc, type_ = SPECS[self.algorithm][k]
            
            if not self.soft_hint or hint_step is None:
                input = preprocess(value, type_, nb_nodes)
            else:
                input = value.unsqueeze(-1) if type_ != Type.CATEGORICAL else value

            encoding = self.encoder[k](input)

            if (loc == Location.NODE and type_ != Type.POINTER) or (loc == Location.GRAPH and type_ == Type.POINTER):
                node_hidden += encoding

            elif loc == Location.EDGE or (loc == Location.NODE and type_ == Type.POINTER):
                if loc == Location.EDGE and type_ == Type.POINTER:
                    encoding2 = self.encoder[k+"_2"](input)

                    edge_hidden += torch.mean(encoding, dim=1) + torch.mean(encoding2, dim=2)
                else:
                    edge_hidden += encoding

            elif loc == Location.GRAPH and type_ != Type.POINTER:
                graph_hidden += encoding

            if loc == Location.NODE and type_ == Type.POINTER:
                input = input.squeeze(-1)
                adj_mat += ((input + input.permute((0, 2, 1))) > 0.5)
                
            elif loc == Location.EDGE and type_ == Type.MASK:
                input = input.squeeze(-1)
                adj_mat += ((input + input.permute((0, 2, 1))) > 0.0)

        return node_hidden, edge_hidden, graph_hidden, (adj_mat > 0.).to(torch.float)

    def forward(self, batch, hints=None, hint_step=None):
        if batch.inputs.batch is not None:
            batch_size = len(batch.inputs.batch)
        else:
            batch_size = 1
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
    

    