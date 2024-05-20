import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from algo_reasoning.src.data.data import CLRSData
from algo_reasoning.src.data.specs import Stage, Location, Type

_Tensor = torch.Tensor

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
