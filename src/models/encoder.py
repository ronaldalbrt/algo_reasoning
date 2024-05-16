import torch.nn as nn
import torch
import torch_scatter
from loguru import logger
## Base encoders and decoders

class NodeBaseEncoder(nn.Module):
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

_ENCODER_MAP = {
    ('node', 'scalar'): NodeBaseEncoder,
    ('node', 'mask'): NodeBaseEncoder,
    ('node', 'mask_one'): NodeBaseEncoder,
}


class Encoder(nn.Module):
    def __init__(self, specs, hidden_dim=128):
        super().__init__()
        self.specs = specs
        self.hidden_dim = hidden_dim
        self.encoder = nn.ModuleDict()
        for k, v in specs.items():
            stage, loc, type_ = v
            if loc == 'edge':
                logger.debug(f'Ignoring edge encoder for {k}')
                continue
            elif stage == 'hint':
                logger.debug(f'Ignoring hint encoder for {k}')
                continue
            elif stage == 'output':
                logger.debug(f'Ignoring output encoder for {k}')
                continue
            else:
                # Input DIM currently hardcoded to 1
                self.encoder[k] = _ENCODER_MAP[(loc, type_)](1, hidden_dim)

    def forward(self, batch):
        hidden = None
        for key, value in batch.inputs:
            if key not in self.encoder:
                logger.debug(f"Ignoring {key}")
                continue
            logger.debug(f"Encoding {key}")
            type(value)
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
