import torch.nn as nn
import torch
from loguru import logger

######################

#TODO: Implement other Processor Architectures

class PGN(nn.Module):
    # TODO: Implement gated message passing
    """Pointer Graph Networks (Veličković et al., NeurIPS 2020)."""
    """Adapted from https://github.com/google-deepmind/clrs/blob/master/clrs/_src/processors.py"""
    def __init__(self, in_channels, out_channels, aggr="max", activation=nn.ReLU(), layer_norm=True, nb_triplet_fts=8):
        super().__init__()
        
        logger.info(f"PGN: in_channels: {in_channels}, out_channels: {out_channels}")
        self.in_channels = in_channels
        self.mid_channels = out_channels
        self.out_channels = out_channels
        self.activation = activation
        self.nb_triplet_fts = nb_triplet_fts
        self.aggr = aggr
        self.layer_norm = layer_norm

        # Message MLPs
        self.m_1 = nn.Linear(in_channels*2, self.mid_channels)
        self.m_2 = nn.Linear(in_channels*2, self.mid_channels)
        self.m_e = nn.Linear(in_channels, self.mid_channels)
        self.m_g = nn.Linear(in_channels, self.mid_channels)

        if self.layer_norm:
            self.norm = nn.LayerNorm(out_channels)
        
        self.msg_mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.mid_channels, self.mid_channels),
            nn.ReLU(),
            nn.Linear(self.mid_channels, self.mid_channels)
        )

        # Output MLP
        self.o1 = nn.Linear(in_channels*2, out_channels) # skip connection
        self.o2 = nn.Linear(self.mid_channels, out_channels)

        if self.nb_triplet_fts is not None:
            self.t_1 = nn.Linear(in_channels*2, nb_triplet_fts)
            self.t_2 = nn.Linear(in_channels*2, nb_triplet_fts)
            self.t_3 = nn.Linear(in_channels*2, nb_triplet_fts)
            self.t_e_1 = nn.Linear(in_channels, nb_triplet_fts)
            self.t_e_2 = nn.Linear(in_channels, nb_triplet_fts)
            self.t_e_3 = nn.Linear(in_channels, nb_triplet_fts)
            self.t_g = nn.Linear(in_channels, nb_triplet_fts)

            self.o3 = nn.Linear(nb_triplet_fts, out_channels)

    def get_triplet_msgs(self, node_fts, edge_fts, graph_fts):
        """Triplet messages, as done by Dudzik and Velickovic (2022)."""
        tri_1 = self.t_1(node_fts)
        tri_2 = self.t_2(node_fts)
        tri_3 = self.t_1(node_fts)
        tri_e_1 = self.t_e_1(edge_fts)
        tri_e_2 = self.t_e_2(edge_fts)
        tri_e_3 = self.t_e_3(edge_fts)
        tri_g = self.t_g(graph_fts)

        return (
            tri_1[:, :, None, None, :]  +  #   (B, N, 1, 1, H)
            tri_2[:, None, :, None, :]  +  # + (B, 1, N, 1, H)
            tri_3[:, None, None, :, :]  +  # + (B, 1, 1, N, H)
            tri_e_1[:, :, :, None, :]   +  # + (B, N, N, 1, H)
            tri_e_2[:, :, None, :, :]   +  # + (B, N, 1, N, H)
            tri_e_3[:, None, :, :, :]   +  # + (B, 1, N, N, H)
            tri_g[:, None, None, None, :]  # + (B, 1, 1, 1, H)
        )                                  # = (B, N, N, N, H)


    def forward(self, node_fts, edge_fts, graph_fts, hidden, adj_matrix):
        z = torch.concat([node_fts, hidden], dim=-1)

        msg_1 = self.m_1(z)
        msg_2 = self.m_2(z)
        msg_e = self.m_e(edge_fts)
        msg_g = self.m_g(graph_fts)

        msgs = msg_1[:, None, :, :] + msg_2[:, :, None, :] + msg_e + msg_g[:, None, None, :] # (B, N, N, H)
        msgs = self.msg_mlp(msgs)
        
        tri_msgs = None
        if self.nb_triplet_fts is not None:
            # Triplet messages, as done by Dudzik and Velickovic (2022)
            triplets = self.get_triplet_msgs(z, edge_fts, graph_fts)

            
            tri_msgs = self.o3(torch.amax(triplets, dim=1))  # (B, N, N, H)
            if self.activation is not None:
                tri_msgs = self.activation(tri_msgs)

            if self.layer_norm:
                tri_msgs = self.norm(tri_msgs)

        if self.aggr == "max":
            maxarg = torch.where(adj_matrix.unsqueeze(-1) == 1, msgs, -1e9) # (B, N, N, H)
            msgs = torch.amax(maxarg, dim=1) # (B, N, H)

        h_1 = self.o1(z)
        h_2 = self.o2(msgs)
        out = h_1 + h_2

        if self.activation is not None:
            out = self.activation(out)

        if self.layer_norm:
            out = self.norm(out)

        return out, tri_msgs


