import torch.nn as nn
import torch
from loguru import logger
from algo_reasoning.src.specs import Stage, Location, Type, SPECS, CATEGORIES_DIMENSIONS
from algo_reasoning.src.data import CLRSData


## TODO: AJUSTAR DECODERS!!!
## Node decoders
class NodeBaseDecoder(nn.Module):
    def __init__(self, spec_dim, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.spec_dim = spec_dim
        self.lin = nn.Linear(hidden_dim, spec_dim)

    def forward(self, x, edge_emb, graph_fts):
        x = self.lin(x).squeeze(-1) # (B, N, H)

        return x
    
class NodePointerDecoder(NodeBaseDecoder):
    def __init__(self, spec_dim, hidden_dim=128):
        super().__init__(hidden_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        self.spec_dim = spec_dim

        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.edge_lin = nn.Linear(hidden_dim, hidden_dim)

        self.lin3 = nn.Linear(hidden_dim, spec_dim)

    def forward(self, x, edge_emb, graph_fts):
        x_1 = super().forward(x, edge_emb, graph_fts)
        x_2 = self.lin2(x)

        edge_x = self.edge_lin(edge_emb)

        x_e = x_2.unsqueeze(-2) + edge_x
        x_m = torch.maximum(x_1.unsqueeze(-2), x_e.permute(0, 2, 1, 3))

        preds = self.lin3(x_m).squeeze(-1) # (B, N, N)

        return preds

#### Edge decoders
class EdgeBaseDecoder(nn.Module):
    def __init__(self, spec_dim, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.spec_dim = spec_dim
        self.lin1 = nn.Linear(hidden_dim, spec_dim)
        self.lin2 = nn.Linear(hidden_dim, spec_dim)
        self.edge_lin = nn.Linear(hidden_dim, spec_dim)

    def forward(self, x, edge_emb, graph_fts):
        x_1 = self.lin1(x) # (B, N, H)
        x_2 = self.lin2(x) # (B, N ,H)

        edge_x = self.edge_lin(edge_emb) #(B, N, N, H)

        out = x_1.unsqueeze(-2) + x_2.unsqueeze(-3) + edge_x  #(B, N, N, H)
        
        return out.squeeze(-1) # (B, N, N)
    
class EdgePointerDecoder(EdgeBaseDecoder):
    def __init__(self, spec_dim, hidden_dim=128):
        super().__init__(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, hidden_dim)
        self.lin4 = nn.Linear(hidden_dim, spec_dim)

    def forward(self, x, edge_emb, graph_fts):
        out = super().forward(x, edge_emb, graph_fts) # (B, N, N)

        pred_2 = self.lin3(x) # (B, N, H)
        p_m = torch.maximum(out.unsqueeze(-2), pred_2.unsqueeze(-3).unsqueeze(-3)) # (B, N, N, H)

        out = self.lin4(p_m).squeeze(-1) # (B, N, N, H)
        
        return out # (B, N, N, N)


#### Graph decoders
class GraphBaseDecoder(nn.Module):
    def __init__(self, spec_dim, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.spec_dim = spec_dim
        self.lin1 = nn.Linear(hidden_dim, spec_dim)
        self.lin2 = nn.Linear(hidden_dim, spec_dim)

    def forward(self, x, edge_emb, graph_fts):
        graph_embedding = torch.max(x, dim=-2) # (B, H)

        pred_n = self.lin1(graph_embedding) # (B, H)
        pred_g = self.lin2(graph_fts) # (B, H)

        out = pred_n + pred_g  # (B, H)

        return out.squeeze(-1)
    
class GraphPointerDecoder(GraphBaseDecoder):
    def __init__(self, spec_dim, hidden_dim=128):
        super().__init__(spec_dim, hidden_dim)

        self.lin3 = nn.Linear(hidden_dim, spec_dim)

    def forward(self, x, edge_emb, graph_fts):
        pred = super().forward(x, edge_emb, graph_fts) # (B, H)

        pred_2 = self.lin3(x) # (B, N, 1)
        ptr_p = pred.unsqueeze(-1) + pred_2  # (B, N, 1)

        out = ptr_p.squeeze(-1)

        return out


_DECODER_MAP = {
    ('node', 'scalar'): NodeBaseDecoder,
    ('node', 'mask'): NodeBaseDecoder,
    ('node', 'mask_one'): NodeBaseDecoder,
    ('node', 'categorical'): NodeBaseDecoder,
    ('node', 'pointer'): NodePointerDecoder,
    ('edge', 'scalar'): EdgeBaseDecoder,
    ('edge', 'mask'): EdgeBaseDecoder,
    ('edge', 'mask_one'): EdgeBaseDecoder,
    ('edge', 'categorical'): EdgeBaseDecoder,
    ('edge', 'pointer'): EdgePointerDecoder,
    ('graph', 'scalar'): GraphBaseDecoder,
    ('graph', 'mask'): GraphBaseDecoder,
    ('graph', 'mask_one'): GraphBaseDecoder,
    ('graph', 'categorical'): GraphBaseDecoder,
    ('graph', 'pointer'): GraphPointerDecoder
}
    
class Decoder(nn.Module):
    def __init__(self, algorithm, hidden_dim=128, decode_hints=True):
        super().__init__()
        self.algorithm = algorithm
        self.hidden_dim = hidden_dim
        self.decode_hints = decode_hints
        self.decoder = nn.ModuleDict()

        self.specs = SPECS[algorithm]
        for k, v in self.specs.items():
            stage, loc, type_ = v

            if (not decode_hints) and stage == 'hint':
                continue
            if stage == 'input':
                continue

            spec_dim = 1
            if type_ == Type.CATEGORICAL:
                spec_dim = CATEGORIES_DIMENSIONS[algorithm][k]

            if k not in self.decoder:
                self.decoder[k] = _DECODER_MAP[(loc, type_)](spec_dim, hidden_dim)

    def forward(self, node_fts, edge_fts, graph_fts):
        outputs = dict()
        hints = dict()
        
        for k, v in self.specs.items():
            stage, loc, type_ = v

            if not self.decode_hints and stage == Stage.HINT:
                continue
            if stage == Stage.INPUT:
                continue
            
            if stage == Stage.OUTPUT:
                outputs[k] = self.decoder[k](node_fts, edge_fts, graph_fts)
            elif stage == Stage.HINT:
                hints[k] = self.decoder[k](node_fts, edge_fts, graph_fts).unsqueeze(1)

        return CLRSData(inputs=CLRSData(), hints=CLRSData().from_dict(hints), length=-1, outputs=CLRSData().from_dict(outputs), algorithm=self.algorithm)
