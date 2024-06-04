import torch.nn as nn
import torch
import torch_scatter
from loguru import logger
from algo_reasoning.src.data.specs import Stage, Location, Type, SPECS, CATEGORIES_DIMENSIONS
from algo_reasoning.src.data.data import CLRSData

## Node decoders
class NodeBaseDecoder(nn.Module):
    def __init__(self, spec_dim, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.spec_dim = spec_dim
        self.lin = nn.Linear(hidden_dim, spec_dim)

    def forward(self, x, *args, **kwargs):
        x = self.lin(x) # (B, N, H)
        return x

class NodeScalarDecoder(NodeBaseDecoder):
    def __init__(self, spec_dim, hidden_dim=128):
        super().__init__(spec_dim, hidden_dim)

    def forward(self, x, *args, **kwargs):
        out = super().forward(x).squeeze(-1) # (B, N)
        return out

class NodeMaskDecoder(NodeBaseDecoder):
    def __init__(self, spec_dim, hidden_dim=128):
        super().__init__(spec_dim, hidden_dim)

    def forward(self, x, *args, **kwargs):
        out = super().forward(x).squeeze(-1) # (B, N)

        return out

class NodeMaskOneDecoder(NodeBaseDecoder):
    def __init__(self, spec_dim, hidden_dim=128):
        super().__init__(spec_dim, hidden_dim)

    def forward(self, x, **kwargs):
        out = super().forward(x).squeeze(-1) # (B, N)

        return out

class NodeCategoricalDecoder(NodeBaseDecoder):
    def __init__(self, spec_dim, hidden_dim=128):
        super().__init__(spec_dim, hidden_dim)

    def forward(self, x, **kwargs):
        out = super().forward(x) # (B, N, C)

        return out
    
class NodePointerDecoder(NodeBaseDecoder):
    def __init__(self, nb_nodes, hidden_dim=128):
        super().__init__(nb_nodes, hidden_dim)

    def forward(self, x, **kwargs):
        out = super().forward(x) # (B, N, N)

        return out



#### Edge decoders
class BaseEdgeDecoder(nn.Module):
    def __init__(self, spec_dim, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.spec_dim = spec_dim
        self.lin = nn.Linear(hidden_dim, spec_dim)

    def forward(self, edge_emb):
        out = self.lin(edge_emb) # (B, N, N, H)
        return out
    
class EdgeScalarDecoder(BaseEdgeDecoder):
    def __init__(self, spec_dim, hidden_dim=128):
        super().__init__(spec_dim, hidden_dim)

    def forward(self, edge_emb, **kwargs):
        out = super().forward(edge_emb).squeeze(-1) # (B, N, N)

        return out
class EdgeMaskDecoder(BaseEdgeDecoder):
    def __init__(self, spec_dim, hidden_dim=128):
        super().__init__(spec_dim, hidden_dim)

    def forward(self, edge_emb, **kwargs):
        out = super().forward(edge_emb).squeeze(-1) # (B, N, N)

        return out
    
class EdgeMaskOneDecoder(BaseEdgeDecoder):
    def __init__(self, spec_dim, hidden_dim=128):
        super().__init__(spec_dim, hidden_dim)

    def forward(self, edge_emb, **kwargs):
        out = super().forward(edge_emb).squeeze(-1) # (B, N, N)

        return out
    
class EdgeCategoricalDecoder(BaseEdgeDecoder):
    def __init__(self, spec_dim, hidden_dim=128):
        super().__init__(spec_dim, hidden_dim)

    def forward(self, edge_emb, **kwargs):
        out = super().forward(edge_emb) # (B, N, N, C)

        return out
    
class EdgePointerDecoder(BaseEdgeDecoder):
    def __init__(self, nb_nodes, hidden_dim=128):
        super().__init__(nb_nodes, hidden_dim)

    def forward(self, edge_emb, **kwargs):
        out = super().forward(edge_emb) # (B, N, N, N)
        
        return out


#### Graph decoders
class GraphBaseDecoder(nn.Module):
    def __init__(self, spec_dim, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.spec_dim = spec_dim
        self.lin = nn.Linear(hidden_dim, spec_dim)

    def forward(self, x, **kwargs):
        x = self.lin(x) # (B, N, H)
        out = torch.mean(x, 1) # (B, H)
        
        return out

class GraphScalarDecoder(GraphBaseDecoder):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__(input_dim, hidden_dim)

    def forward(self, x, **kwargs):
        out = super().forward(x).squeeze(-1) # (B)
        
class GraphMaskDecoder(GraphBaseDecoder):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__(input_dim, hidden_dim)

    def forward(self, x, **kwargs):
        out = super().forward(x).squeeze(-1) # (B)
        
        return out

class GraphMaskOneDecoder(GraphBaseDecoder):
    def __init__(self, spec_dim, hidden_dim=128):
        super().__init__(spec_dim, hidden_dim)

    def forward(self, x, **kwargs):
        out = super().forward(x).squeeze(-1) # (B)
        
        return out

class GraphCategoricalDecoder(GraphBaseDecoder):
    def __init__(self, spec_dim, hidden_dim=128):
        super().__init__(spec_dim, hidden_dim)

    def forward(self, x, **kwargs):
        out = super().forward(x) # (B, C)

        return out
    
class GraphPointerDecoder(GraphBaseDecoder):
    def __init__(self, nb_nodes, hidden_dim=128):
        super().__init__(nb_nodes, hidden_dim)

    def forward(self, x, **kwargs):
        out = super().forward(x) # (B, N)

        return out
    

_DECODER_MAP = {
    ('node', 'scalar'): NodeScalarDecoder,
    ('node', 'mask'): NodeMaskDecoder,
    ('node', 'mask_one'): NodeMaskOneDecoder,
    ('node', 'categorical'): NodeCategoricalDecoder,
    ('node', 'pointer'): NodePointerDecoder,
    ('edge', 'scalar'): EdgeScalarDecoder,
    ('edge', 'mask'): EdgeMaskDecoder,
    ('edge', 'mask_one'): EdgeMaskOneDecoder,
    ('edge', 'categorical'): EdgeCategoricalDecoder,
    ('edge', 'pointer'): EdgePointerDecoder,
    ('graph', 'scalar'): GraphScalarDecoder,
    ('graph', 'mask'): GraphMaskDecoder,
    ('graph', 'mask_one'): GraphMaskOneDecoder,
    ('graph', 'categorical'): GraphCategoricalDecoder,
    ('graph', 'pointer'): GraphPointerDecoder
}
    
class Decoder(nn.Module):
    def __init__(self, algorithm, nb_nodes=16, hidden_dim=128, no_hint=False):
        super().__init__()
        self.algorithm = algorithm
        self.hidden_dim = hidden_dim
        self.no_hint = no_hint
        self.decoder = nn.ModuleDict()

        self.specs = SPECS[algorithm]
        for k, v in self.specs.items():
            stage, loc, type_ = v
            cat_dim = CATEGORIES_DIMENSIONS[algorithm][k]

            if no_hint and stage == 'hint':
                logger.debug(f'Ignoring hint decoder for {k}')
                continue
            if stage == 'input':
                logger.debug(f'Ignoring input decoder for {k}')
                continue

            spec_dim = 1
            if type_ == Type.CATEGORICAL:
                spec_dim = cat_dim
            elif type_ == Type.POINTER:
                spec_dim = nb_nodes

            if k not in self.decoder:
                self.decoder[k] = _DECODER_MAP[(loc, type_)](spec_dim, hidden_dim)

    def forward(self, hidden):
        outputs = CLRSData()
        hints = CLRSData()
        
        for k, v in self.specs.items():
            stage, loc, type_ = v

            if self.no_hint and stage == 'hint':
                continue
            if stage == 'input':
                continue
            
            if stage == 'output':
                outputs[k] = self.decoder[k](hidden)
            elif stage == 'hint':
                hints[k] = self.decoder[k](hidden)

        return CLRSData(inputs=CLRSData(), hints=hints, length=None, outputs=outputs, algorithm=self.algorithm)

    
def grab_outputs(hints, batch):
    """This function grabs the outputs from the batch and returns them in the same format as the hints"""
    output = {}
    for k in hints:
        k_out = k.replace('_h', '')
        if k_out in batch.outputs:
            output[k_out] = hints[k]
    return output

def output_mask(batch, step):
    final_node_idx = (batch.length[batch.batch]-1)

    masks = {}
    for key in batch.outputs:
        if key in batch.edge_attrs():
            final_edge_idx = final_node_idx[batch.edge_index[0]]
            masks[key] = final_edge_idx == step
        elif key in batch.node_attrs():
            masks[key] = final_node_idx == step
        else:
            # graph attribute
            masks[key] = batch.length == step + 1
    return masks