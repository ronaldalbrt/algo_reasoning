import torch
import torch.nn as nn
import torch.nn.functional as F

import math

######################

#TODO: Implement other Processor Architectures

class PGN(nn.Module):
    """Pointer Graph Networks (Veličković et al., NeurIPS 2020)."""
    """Adapted from https://github.com/google-deepmind/clrs/blob/master/clrs/_src/processors.py"""
    def __init__(self, in_size, out_size, aggr="max", activation=nn.ReLU(), layer_norm=True, nb_triplet_fts=8, gated=False):
        super().__init__()
        
        self.in_size = in_size
        self.mid_channels = out_size
        self.out_size = out_size
        self.gated = gated
        self.activation = activation
        self.nb_triplet_fts = nb_triplet_fts
        self.aggr = aggr
        self.layer_norm = layer_norm

        # Message MLPs
        self.m_1 = nn.Linear(in_size*2, self.mid_channels)
        self.m_2 = nn.Linear(in_size*2, self.mid_channels)
        self.m_e = nn.Linear(in_size, self.mid_channels)
        self.m_g = nn.Linear(in_size, self.mid_channels)

        if self.layer_norm:
            self.norm = nn.LayerNorm(out_size)
        
        self.msg_mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.mid_channels, self.mid_channels),
            nn.ReLU(),
            nn.Linear(self.mid_channels, self.mid_channels)
        )

        # Output MLP
        self.o1 = nn.Linear(in_size*2, out_size) # skip connection
        self.o2 = nn.Linear(self.mid_channels, out_size)

        if self.nb_triplet_fts is not None:
            self.t_1 = nn.Linear(in_size*2, nb_triplet_fts)
            self.t_2 = nn.Linear(in_size*2, nb_triplet_fts)
            self.t_3 = nn.Linear(in_size*2, nb_triplet_fts)
            self.t_e_1 = nn.Linear(in_size, nb_triplet_fts)
            self.t_e_2 = nn.Linear(in_size, nb_triplet_fts)
            self.t_e_3 = nn.Linear(in_size, nb_triplet_fts)
            self.t_g = nn.Linear(in_size, nb_triplet_fts)
            self.o3 = nn.Linear(nb_triplet_fts, out_size)

        if self.gated:
            self.gate1 = nn.Linear(in_size*2, out_size)
            self.gate2 = nn.Linear(self.mid_channels, out_size)
            self.gate3 = nn.Linear(out_size, out_size)

        self.reset_parameters()

    def reset_parameters(self):
        if self.gated:
            nn.init.constant_(self.gate3.weight, -3)

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

        if self.gated:
            gate = F.sigmoid(self.gate3(F.relu(self.gate1(z) + self.gate2(msgs))))
            out = out * gate + hidden * (1-gate)

        return out, tri_msgs

class GAT(nn.Module):
    def __init__(self, in_size, out_size, nb_heads=8, activation=nn.ReLU(), layer_norm=True, residual=True, nb_triplet_fts=8) -> None:
        super().__init__()
        self.nb_heads = nb_heads
        self.head_dim = out_size//nb_heads
        self.nb_triplet_fts = nb_triplet_fts
        self.activation = activation
        self.layer_norm = layer_norm
        self.residual = residual

        if self.residual:
            self.skip = nn.Linear(in_size*2, out_size)
        if self.layer_norm:
            self.norm = nn.LayerNorm(out_size)
        
        self.m = nn.Linear(in_size*2, out_size, bias=False)

        self.a_1 = nn.Linear(in_size*2, nb_heads, bias=False)
        self.a_2 = nn.Linear(in_size*2, nb_heads, bias=False)
        self.a_e = nn.Linear(in_size, nb_heads, bias=False)
        self.a_g = nn.Linear(in_size, nb_heads, bias=False) 

        self.node_ffn = nn.Sequential(nn.Linear(out_size, out_size), nn.ReLU(), nn.Linear(out_size, out_size), nn.ReLU())

        if self.nb_triplet_fts is not None:
            self.t_1 = nn.Linear(in_size*2, nb_triplet_fts)
            self.t_2 = nn.Linear(in_size*2, nb_triplet_fts)
            self.t_3 = nn.Linear(in_size*2, nb_triplet_fts)
            self.t_e_1 = nn.Linear(in_size, nb_triplet_fts)
            self.t_e_2 = nn.Linear(in_size, nb_triplet_fts)
            self.t_e_3 = nn.Linear(in_size, nb_triplet_fts)
            self.t_g = nn.Linear(in_size, nb_triplet_fts)
            self.o3 = nn.Linear(nb_triplet_fts, out_size)


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

        batch_size, nb_nodes, _ = hidden.size()

        values = self.m(z)
        values = values.view(batch_size, nb_nodes, self.nb_heads, self.head_dim).transpose(1, 2)
        
        att_1 = self.a_1(z).unsqueeze(-1)
        att_2 = self.a_2(z).unsqueeze(-1)
        att_e = self.a_e(edge_fts)
        att_g = self.a_g(graph_fts).unsqueeze(-1)

        logits = att_1.permute(0, 2, 1, 3) + att_2.permute(0, 2, 3, 1) + att_e.permute(0, 3, 1, 2) + att_g.unsqueeze(-1)
        coefs = adj_matrix.unsqueeze(1) * torch.softmax(F.leaky_relu(logits), dim=-1)

        out = torch.matmul(coefs, values)
        out = out.permute(0, 2, 1, 3).reshape(batch_size, nb_nodes, -1)  

        if self.residual:
            out += self.skip(z)

        if self.activation is not None:
            out = self.activation(out)

        if self.layer_norm:
            out = self.norm(out)

        tri_msgs = None
        if self.nb_triplet_fts is not None:
            # Triplet messages, as done by Dudzik and Velickovic (2022)
            triplets = self.get_triplet_msgs(z, edge_fts, graph_fts)
            
            tri_msgs = self.o3(torch.amax(triplets, dim=1))  # (B, N, N, H)
            if self.activation is not None:
                tri_msgs = self.activation(tri_msgs)

            if self.layer_norm:
                tri_msgs = self.norm(tri_msgs)

        return out, tri_msgs

class FullGAT(GAT):
    def forward(self, node_fts, edge_fts, graph_fts, hidden, adj_mat):
        adj_mat = torch.ones_like(adj_mat)
        return super().forward(node_fts, edge_fts, graph_fts, hidden, adj_mat)


class MPNN(PGN):
    def forward(self, node_fts, edge_fts, graph_fts, hidden, adj_mat):
        adj_mat = torch.ones_like(adj_mat)
        return super().forward(node_fts, edge_fts, graph_fts, hidden, adj_mat)