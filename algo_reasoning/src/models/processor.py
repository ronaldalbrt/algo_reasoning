# Copyright (C) 2024 Ronald Albert ronaldalbert@cos.ufrj.br

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#         http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

def spectral_decomposition(adj_matrix):
    degrees = torch.sum(adj_matrix, dim=1)
    degree_matrix = torch.stack([torch.diag(degrees[d]) for d in range(degrees.size(0))], dim=0)
    laplacian = degree_matrix - adj_matrix
    
    result = torch.linalg.eigh(laplacian)
    eigenvalues = result.eigenvalues
    eigenvectors = result.eigenvectors

    return eigenvectors, eigenvalues

def normalized_laplacian(adj_matrix):
    degrees = torch.sum(adj_matrix, dim=1)
    degrees_inv_sqrt = torch.where(degrees == 0, 1, torch.rsqrt(degrees))

    degrees_inv_sqrt = torch.stack([torch.diag(degrees_inv_sqrt[d]) for d in range(degrees.size(0))], dim=0)

    # A_{sym} = D^{-0.5} * A * D^{-0.5}
    normalized_adj = degrees_inv_sqrt@adj_matrix@degrees_inv_sqrt
    normalized_laplacian = torch.eye(adj_matrix.size(1), device=adj_matrix.device).unsqueeze(0) - normalized_adj

    return normalized_laplacian

def largest_singularvalue(A, n_it=1000):
    curr_x = torch.randn((A.size(0), A.size(-1)), device=A.device)
    curr_x = curr_x/torch.norm(curr_x,  dim=-1, keepdim=True)

    prev_x = torch.zeros((A.size(0), A.size(-1)), device=A.device)
    for i in range(n_it): 
        prev_x = torch.clone(curr_x)
         
        curr_x = torch.bmm(A, curr_x.unsqueeze(-1)).squeeze()
        curr_x = curr_x/torch.norm(curr_x, dim=-1, keepdim=True)



    return torch.norm(torch.bmm(A, curr_x.unsqueeze(-1)).squeeze(),  dim=-1)


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
    """Graph Attention Network (Velickovic et al., ICLR 2018)."""
    """Adapted from https://github.com/google-deepmind/clrs/blob/master/clrs/_src/processors.py"""
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

class DeepSetsLayer(nn.Module):
    def __init__(self, in_size, out_size):
        super(DeepSetsLayer, self).__init__()
        self.lin1 = nn.Linear(in_size, in_size)
        self.lin2 = nn.Linear(in_size, out_size)
        self.act = nn.ReLU()

    def forward(self, x):
        xm, _ = x.max(1, keepdim=True)
        xm = self.lin1(xm) 
        x = self.lin2(x)
        x = x - xm

        return self.act(x)

class SpecFormerConv(nn.Module):
    def __init__(self, hidden_size):
        super(SpecFormerConv, self).__init__()

        self.pre_ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU()
        )

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        )

    def forward(self, node_fts, edge_fts, bases):
        edge_fts = self.pre_ffn(edge_fts) * bases

        edge_fts_sum = torch.sum(edge_fts, dim=1)

        node_fts = node_fts + edge_fts_sum
        y = self.ffn(node_fts)
        node_fts = node_fts + y

        return node_fts, edge_fts

class SpectralMPNN(nn.Module):
    def __init__(self, in_size, out_size, 
                activation=nn.ReLU(), 
                layer_norm=True,
                nb_heads=8, 
                *args, **kwargs):   
        super().__init__()

        self.in_size = in_size
        self.mid_channels = out_size
        self.out_size = out_size
        self.activation = activation
        self.layer_norm = layer_norm
        self.nb_heads = nb_heads

        if self.layer_norm:
            self.norm = nn.LayerNorm(out_size)

        self.m_1 = nn.Linear(in_size*2, self.mid_channels)
        self.m_2 = nn.Linear(in_size*2, self.mid_channels)
        self.m_e = nn.Linear(in_size, self.mid_channels)
        self.m_g = nn.Linear(in_size, self.mid_channels)

        self.msg_mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.mid_channels, self.mid_channels),
            nn.ReLU(),
            nn.Linear(self.mid_channels, out_size)
        )

        self.msg_proj = nn.Linear(out_size, nb_heads)

        self.feat_encoder = nn.Sequential(
            nn.Linear(in_size*2, out_size),
            nn.ReLU(),
            nn.Linear(out_size, out_size),
        )

        self.node_proj = nn.Linear(nb_heads, 1)

        self.filter_encoder = nn.Sequential(
            nn.Linear(nb_heads + 1, out_size),
            nn.LayerNorm(out_size),
            nn.GELU(),
            nn.Linear(out_size, out_size),
            nn.LayerNorm(out_size),
            nn.GELU(),
        )

        self.layers = nn.ModuleList([SpecFormerConv(out_size) for i in range(1)])

        self.o1 = nn.Linear(out_size, out_size)
        self.o2 = nn.Linear(out_size, out_size)
        self.o3 = nn.Linear(2*in_size, out_size)


    def spectral_decomposition(self, adj_matrix):
        degrees = torch.sum(adj_matrix, dim=1)
        degree_matrix = torch.stack([torch.diag(degrees[d]) for d in range(degrees.size(0))], dim=0)
        laplacian = degree_matrix - adj_matrix
        
        result = torch.linalg.eigh(laplacian)
        eigenvalues = result.eigenvalues
        eigenvectors = result.eigenvectors

        return eigenvectors, eigenvalues
    
    def forward(self, node_fts, edge_fts, graph_fts, hidden, adj_matrix):
        z = torch.concat([node_fts, hidden], dim=-1)
        eig_vectors, eig_values = self.spectral_decomposition(adj_matrix)

        h = self.feat_encoder(z)
        
        msg_1 = self.m_1(z)
        msg_2 = self.m_2(z)
        msg_e = self.m_e(edge_fts)
        msg_g = self.m_g(graph_fts)

        msgs = msg_1[:, None, :, :] + msg_2[:, :, None, :] + msg_e + msg_g[:, None, None, :] # (B, N, N, H)
        msgs = self.msg_mlp(msgs)
        msgs = torch.amax(msgs, dim=1)

        msgs_proj = self.msg_proj(msgs)
        
        diag_e = torch.diag_embed(msgs_proj.transpose(-2, -1))
        identity = torch.diag_embed(torch.ones_like(eig_values))
        bases = [identity]
        for i in range(self.nb_heads):
            filters = eig_vectors @ diag_e[:, i, :, :] @ eig_vectors.transpose(-2, -1)
            bases.append(filters)

        bases = torch.stack(bases, axis=-1) 
        bases =  self.filter_encoder(bases)
        bases = adj_matrix.unsqueeze(-1) * torch.softmax(bases, dim=-1)

        for conv in self.layers:
            h, edge_fts = conv(h, edge_fts, bases)

        out = self.o1(msgs) + self.o2(h) 

        out = out + self.o3(z)

        if self.layer_norm:
            out = self.norm(out)

        return out, edge_fts

class ChebyshevGraphConv(nn.Module):  
    def __init__(self, in_size, out_size, K = 3, eps=1e-05, layer_norm=True):
        super(ChebyshevGraphConv, self).__init__()
        self.K = K
        self.in_size = in_size
        self.mid_channels = out_size
        self.out_size = out_size
        self.eps = eps
        self.layer_norm = layer_norm

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
            nn.Linear(self.mid_channels, out_size)
        )

        self.node_weights = nn.Parameter(torch.FloatTensor(K, in_size, out_size))
        self.edge_weights = nn.Parameter(torch.FloatTensor(K, in_size, out_size))

        self.feat_encoder = nn.Sequential(
            nn.Linear(in_size*2, out_size),
            nn.ReLU(),
            nn.Linear(out_size, out_size),
        )

        self.o1 = nn.Linear(out_size, out_size)
        self.o2 = nn.Linear(out_size, out_size)
        self.o3 = nn.Linear(2*in_size, out_size)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.node_weights, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.edge_weights, a=math.sqrt(5))

    def forward(self, node_fts, edge_fts, graph_fts, hidden, adj_matrix):
        z = torch.concat([node_fts, hidden], dim=-1)

        lap = normalized_laplacian(adj_matrix)
        eigval_max = torch.linalg.matrix_norm(lap, ord=2)
        cheb_lap = (2 * lap / (eigval_max[:, None, None] + self.eps)) - torch.eye(adj_matrix.size(1), device=adj_matrix.device).unsqueeze(0)    

        h = self.feat_encoder(z)

        msg_1 = self.m_1(z)
        msg_2 = self.m_2(z)
        msg_e = self.m_e(edge_fts)
        msg_g = self.m_g(graph_fts)

        msgs = msg_1[:, None, :, :] + msg_2[:, :, None, :] + msg_e + msg_g[:, None, None, :] # (B, N, N, H)
        msgs = self.msg_mlp(msgs)
        msgs = torch.amax(msgs, dim=1)

        cheb_node_feat = []
        cheb_edge_feat = []

        cheb_node_feat.append(h)
        cheb_edge_feat.append(edge_fts)

        if self.K > 0:
            cheb_node_feat.append(torch.bmm(cheb_lap, h)) # B x N x D
            cheb_edge_feat.append(cheb_lap.unsqueeze(-1) * edge_fts) # B x N x N x D 
            
            for k in range(2, self.K):
                cheb_node_feat.append(torch.bmm(2 * cheb_lap, cheb_node_feat[k - 1]) - cheb_node_feat[k - 2])
                cheb_edge_feat.append(((2 * cheb_lap.unsqueeze(-1)) * cheb_edge_feat[k - 1]) - cheb_edge_feat[k - 2])
                

        cheb_node_feat = torch.stack(cheb_node_feat, dim=1)
        cheb_edge_feat = torch.stack(cheb_edge_feat, dim=1)

        node_out = torch.einsum('bnij,bnij->bij', cheb_node_feat, torch.einsum('bij,njk->bnik', msgs, self.node_weights))
        edge_out = torch.einsum('bnijl,nlk->bijk', cheb_edge_feat, self.edge_weights)
        
        out = self.o1(msgs) + self.o2(node_out)

        out = out + self.o3(z)

        if self.layer_norm:
            out = self.norm(out)

        return out, edge_out