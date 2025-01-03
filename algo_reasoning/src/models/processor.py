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

class SpecFormer(nn.Module):
    def __init__(self, in_size, out_size, nb_heads=8, n_layers=1, activation=nn.ReLU(), layer_norm=True, eig_constant=100):   
        super().__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.nb_heads = nb_heads
        self.activation = activation
        self.layer_norm = layer_norm
        self.constant = eig_constant
        self.n_layers = n_layers


        if self.layer_norm:
            self.norm = nn.LayerNorm(out_size)
            self.mha_norm = nn.LayerNorm(out_size)
            self.ffn_norm = nn.LayerNorm(out_size)

        self.eig_w = nn.Linear(out_size + 1, out_size)
        self.mha = nn.MultiheadAttention(out_size, nb_heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(out_size, out_size), nn.ReLU(), nn.Linear(out_size, out_size))

        self.decoder = nn.Linear(1, nb_heads)

        self.layers = nn.ModuleList([SpecFormerConv(out_size) for i in range(n_layers)])

        self.feat_encoder = nn.Sequential(
            nn.Linear(in_size*2, out_size),
            nn.ReLU(),
            nn.Linear(out_size, out_size),
        )

        self.filter_encoder = nn.Sequential(
            nn.Linear(nb_heads + 1, out_size),
            nn.LayerNorm(out_size),
            nn.GELU(),
            nn.Linear(out_size, out_size),
            nn.LayerNorm(out_size),
            nn.GELU(),
        )

    def spectral_decomposition(self, adj_matrix):
        degrees = torch.sum(adj_matrix, dim=1)
        degree_matrix = torch.stack([torch.diag(degrees[d]) for d in range(degrees.size(0))], dim=0)
        laplacian = degree_matrix - adj_matrix
        
        result = torch.linalg.eigh(laplacian)
        eigenvalues = result.eigenvalues
        eigenvectors = result.eigenvectors

        return eigenvectors, eigenvalues
    
    def eig_encoder(self, e):
        ee = e * self.constant
        div = torch.exp(torch.arange(0, self.out_size, 2)/self.out_size * -math.log(10000)).to(e.device)
        
        pe = ee.unsqueeze(-1) * div
        eeig = torch.cat((e.unsqueeze(-1), torch.sin(pe), torch.cos(pe)), dim=-1)

        return self.eig_w(eeig)
    
    def forward(self, node_fts, edge_fts, graph_fts, hidden, adj_mat):
        z = torch.concat([node_fts, hidden], dim=-1)
        eig_vectors, eig_values = self.spectral_decomposition(adj_mat)

        h = self.feat_encoder(z)

        eig = self.eig_encoder(eig_values)

        mha_eig = self.mha_norm(eig)
        mha_eig, _ = self.mha(mha_eig, mha_eig, mha_eig)
        eig = eig + mha_eig

        ffn_eig = self.ffn_norm(eig)
        ffn_eig = self.ffn(ffn_eig)
        eig = eig + ffn_eig

        new_e = self.decoder(eig_values.unsqueeze(-1)).transpose(2, 1)

        diag_e = torch.diag_embed(new_e)

        identity = torch.diag_embed(torch.ones_like(eig_values))
        bases = [identity]
        for i in range(self.nb_heads):
            filters = eig_vectors @ diag_e[:, i, :, :] @ eig_vectors.transpose(-2, -1)
            bases.append(filters)

        bases = torch.stack(bases, axis=-1) 
        bases = self.filter_encoder(bases)
        bases = adj_mat.unsqueeze(-1) * torch.softmax(bases, dim=-1)

        for conv in self.layers:
            h, edge_fts = conv(h, edge_fts, bases)

        return h, edge_fts

class SpectralFilter(nn.Module):
    def __init__(self, in_size, out_size):   
        super().__init__()

        self.att_1 = nn.Linear(in_size, out_size)
        self.att_2 = nn.Linear(in_size, out_size)

        self.out_lin = nn.Linear(in_size, out_size)

    
    def forward(self, z, eig_values, eig_vectors):
        z_q = self.att_1(z)
        z_k = self.att_2(z)

        attn_weights = torch.matmul(z_q, z_k.transpose(1, 2))
        coefs = torch.softmax(attn_weights, dim=-1)

        filtered_eig = torch.diag_embed(torch.bmm(coefs, eig_values.unsqueeze(-1)).squeeze())

        out = self.out_lin(eig_vectors@(filtered_eig@(eig_vectors.transpose(-2, -1)@z)))
        
        return out

class MLP(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(MLP, self).__init__()
        self.lin1 = nn.Linear(in_size, in_size)
        self.lin2 = nn.Linear(in_size, out_size)
        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.lin1(x))
        x = self.dropout(x)
        return self.lin2(x)
    
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
    
class S2GNN(nn.Module):
    """S2GNN (Simon Geisler, et al. (2024). )"""
    """
        The model implemented is ideally the same as the one in the paper, but the implementation is slightly different 
        from the one proposed, so as to better match Neural Algorithmic Alignment.
    """
    def __init__(self, in_size, out_size, activation=nn.ReLU(), layer_norm=True, fourier_equivariance=True, processor=MPNN, nb_triplet_fts=8, *args, **kwargs):   
        super().__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.activation = activation
        self.layer_norm = layer_norm

        if self.layer_norm:
            self.norm = nn.LayerNorm(out_size)

        self.nodes_proj = nn.Sequential(nn.Linear(2*in_size, in_size), nn.ReLU())
        self.edges_proj = nn.Linear(in_size, in_size)
        self.graph_proj = nn.Linear(in_size, in_size)

        if fourier_equivariance:
            self.fourier_layer = DeepSetsLayer(in_size, out_size)
            self.fourier_edges_layer = DeepSetsLayer(in_size, out_size)
        else:
            self.fourier_layer = nn.Linear(in_size, out_size)
            self.fourier_edges_layer = nn.Linear(in_size, out_size)

        self.node_out = nn.Linear(2*in_size, in_size)
        self.edge_out = nn.Linear(2*in_size, in_size)

        self.processor = processor(in_size, out_size, nb_triplet_fts=nb_triplet_fts, *args, **kwargs)

    def spectral_decomposition(self, adj_matrix):
        degrees = torch.sum(adj_matrix, dim=1)
        degree_matrix = torch.stack([torch.diag(degrees[d]) for d in range(degrees.size(0))], dim=0)
        laplacian = degree_matrix - adj_matrix
        
        result = torch.linalg.eigh(laplacian)
        eigenvalues = result.eigenvalues
        eigenvectors = result.eigenvectors

        return eigenvectors, eigenvalues
    
    def forward(self, node_fts, edge_fts, graph_fts, hidden, adj_mat):
        z = torch.concat([node_fts, hidden], dim=-1)

        msg, tri_msgs = self.processor(node_fts, edge_fts, graph_fts, hidden, adj_mat)

        z = self.nodes_proj(z)
        edge_z = self.edges_proj(edge_fts)

        eig_vectors, _ = self.spectral_decomposition(adj_mat)

        fourier_z = eig_vectors.transpose(-2, -1)@z

        z = torch.concat([self.fourier_layer(fourier_z), msg], dim=-1)

        fourier_edges = (eig_vectors.transpose(-2, -1)@edge_z.transpose(0, 1)).transpose(0, 1)

        edge_z = torch.concat([self.fourier_edges_layer(fourier_edges), tri_msgs], dim=-1)

        return self.node_out(z), self.edge_out(edge_z)

class MLP(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(MLP, self).__init__()
        self.lin1 = nn.Linear(in_size, in_size)
        self.lin2 = nn.Linear(in_size, out_size)
        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.lin1(x))
        x = self.dropout(x)
        return self.lin2(x)

class gfNN(nn.Module):
    def __init__(self, in_size, out_size, activation=nn.ReLU(), layer_norm=True):   
        super().__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.activation = activation
        self.layer_norm = layer_norm

        if self.layer_norm:
            self.norm = nn.LayerNorm(out_size)

        self.nodes_proj = nn.Sequential(nn.Linear(2*in_size, in_size), nn.ReLU())
        self.edges_proj = nn.Linear(in_size, in_size)
        self.graph_proj = nn.Linear(in_size, in_size)

        self.deepsets = DeepSetsLayer(in_size, out_size)
        self.edges_deepsets = DeepSetsLayer(in_size, out_size)

        self.mlp = DeepSetsLayer(in_size, out_size)
        self.edges_mlp = DeepSetsLayer(in_size, out_size)

    
    def forward(self, node_fts, edge_fts, graph_fts, hidden, adj_mat):
        z = torch.concat([node_fts, hidden], dim=-1)

        z = self.nodes_proj(z)
        edge_fts = self.edges_proj(edge_fts)
        graph_fts = self.graph_proj(graph_fts).unsqueeze(1)

        eig_vectors, _ = spectral_decomposition(adj_mat)
        fourier_z = eig_vectors.transpose(-2, -1)@z

        z = self.deepsets(fourier_z + graph_fts) + self.mlp(z)

        fourier_edges = (eig_vectors.transpose(-2, -1)@edge_fts.transpose(0, 1)).transpose(0, 1) + graph_fts.unsqueeze(1)
        edge_fts = self.edges_deepsets(fourier_edges) + self.edges_mlp(edge_fts)

        return z, edge_fts

class SpectralMPNN(nn.Module):
    def __init__(self, in_size, out_size, 
                activation=nn.ReLU(), 
                layer_norm=True,
                nb_triplet_fts=8, 
                gated=True,
                message_passing='mpnn',
                *args, **kwargs):   
        super().__init__()

        self.in_size = in_size
        self.mid_channels = out_size
        self.out_size = out_size
        self.activation = activation
        self.layer_norm = layer_norm
        self.gated = gated
        self.nb_triplet_fts = nb_triplet_fts

        if message_passing == 'mpnn':
            self.mpnn = MPNN(in_size, out_size, activation=activation, layer_norm=layer_norm, nb_triplet_fts=nb_triplet_fts, *args, **kwargs)
        elif message_passing == 'pgn':
            self.mpnn = PGN(in_size, out_size, activation=activation, layer_norm=layer_norm, nb_triplet_fts=nb_triplet_fts, *args, **kwargs)

        self.specformer = SpecFormer(in_size, out_size, activation=activation, layer_norm=layer_norm)

        self.out_mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(2*out_size, out_size),
            nn.ReLU(),
            nn.Linear(out_size, out_size)
        )

        self.edges_mlps = nn.Sequential(
            nn.ReLU(),
            nn.Linear(2*out_size, out_size),
            nn.ReLU(),
            nn.Linear(out_size, out_size)
        )

    def forward(self, node_fts, edge_fts, graph_fts, hidden, adj_matrix):
        mpnn_out, tri_msgs = self.mpnn(node_fts, edge_fts, graph_fts, hidden, adj_matrix)
        specformer_out, edge_out = self.specformer(node_fts, edge_fts, graph_fts, hidden, adj_matrix)

        out = self.out_mlp(torch.concat([mpnn_out, specformer_out], dim=-1))
        edge_out = self.edges_mlps(torch.concat([edge_out, tri_msgs], dim=-1))

        return out, edge_out

class SpectralMPNN2(nn.Module):
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

        self.eig_mlp = nn.Sequential(
            nn.Linear(1, nb_heads),
            nn.ReLU(),
            nn.Linear(nb_heads, nb_heads),
            nn.ReLU()
        )

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

        new_e = self.eig_mlp(eig_values.unsqueeze(-1))
        new_e = self.msg_proj(msgs) * new_e
        
        diag_e = torch.diag_embed(new_e.transpose(-2, -1))

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

        return out, edge_fts