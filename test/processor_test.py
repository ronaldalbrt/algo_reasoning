import unittest
import torch
import torch.nn as nn
from algo_reasoning.src.models.processor import (
    PGN, GAT, FullGAT, MPNN, DeepSetsLayer, 
    SpecFormerConv, SpectralMPNN, PolynomialSpectralMPNN,
    spectral_decomposition, normalized_laplacian, largest_singularvalue
)

class ProcessorTest(unittest.TestCase):
    def setUp(self):
        # Common test parameters
        self.batch_size = 2
        self.nb_nodes = 4
        self.hidden_dim = 8
        self.edge_dim = 8
        self.graph_dim = 8
        self.nb_triplet_fts = 4
        
        # Create sample inputs
        self.node_fts = torch.randn(self.batch_size, self.nb_nodes, self.hidden_dim)
        self.edge_fts = torch.randn(self.batch_size, self.nb_nodes, self.nb_nodes, self.edge_dim)
        self.graph_fts = torch.randn(self.batch_size, self.graph_dim)
        self.hidden = torch.randn(self.batch_size, self.nb_nodes, self.hidden_dim)
        
        # Create adjacency matrix (fully connected for testing)
        self.adj_matrix = torch.ones(self.batch_size, self.nb_nodes, self.nb_nodes)
        # Set diagonal to 1 (self-connections)
        for i in range(self.batch_size):
            self.adj_matrix[i].fill_diagonal_(1)
    
    def test_utility_functions(self):
        """Test the utility functions for spectral methods."""
        # Test spectral_decomposition
        eigenvectors, eigenvalues = spectral_decomposition(self.adj_matrix)
        self.assertEqual(eigenvectors.shape, (self.batch_size, self.nb_nodes, self.nb_nodes))
        self.assertEqual(eigenvalues.shape, (self.batch_size, self.nb_nodes))
        
        # Test normalized_laplacian
        norm_lap = normalized_laplacian(self.adj_matrix)
        self.assertEqual(norm_lap.shape, (self.batch_size, self.nb_nodes, self.nb_nodes))
        
        # Test largest_singularvalue
        sing_val = largest_singularvalue(self.adj_matrix, n_it=10)
        self.assertEqual(sing_val.shape, (self.batch_size,))
    
    def test_pgn(self):
        """Test Pointer Graph Networks."""
        model = PGN(self.hidden_dim, self.hidden_dim, nb_triplet_fts=self.nb_triplet_fts)
        out, tri_msgs = model(self.node_fts, self.edge_fts, self.graph_fts, self.hidden, self.adj_matrix)
        
        # Check output shapes
        self.assertEqual(out.shape, (self.batch_size, self.nb_nodes, self.hidden_dim))
        self.assertEqual(tri_msgs.shape, (self.batch_size, self.nb_nodes, self.nb_nodes, self.hidden_dim))
        
        # Test without triplet features
        model_no_triplet = PGN(self.hidden_dim, self.hidden_dim, nb_triplet_fts=None)
        out_no_triplet, tri_msgs_no_triplet = model_no_triplet(
            self.node_fts, self.edge_fts, self.graph_fts, self.hidden, self.adj_matrix
        )
        self.assertEqual(out_no_triplet.shape, (self.batch_size, self.nb_nodes, self.hidden_dim))
        self.assertIsNone(tri_msgs_no_triplet)
        
        # Test with gating
        model_gated = PGN(self.hidden_dim, self.hidden_dim, nb_triplet_fts=self.nb_triplet_fts, gated=True)
        out_gated, _ = model_gated(self.node_fts, self.edge_fts, self.graph_fts, self.hidden, self.adj_matrix)
        self.assertEqual(out_gated.shape, (self.batch_size, self.nb_nodes, self.hidden_dim))
    
    def test_gat(self):
        """Test Graph Attention Networks."""
        model = GAT(self.hidden_dim, self.hidden_dim, nb_triplet_fts=self.nb_triplet_fts)
        out, tri_msgs = model(self.node_fts, self.edge_fts, self.graph_fts, self.hidden, self.adj_matrix)
        
        # Check output shapes
        self.assertEqual(out.shape, (self.batch_size, self.nb_nodes, self.hidden_dim))
        self.assertEqual(tri_msgs.shape, (self.batch_size, self.nb_nodes, self.nb_nodes, self.hidden_dim))
        
        # Test without triplet features
        model_no_triplet = GAT(self.hidden_dim, self.hidden_dim, nb_triplet_fts=None)
        out_no_triplet, tri_msgs_no_triplet = model_no_triplet(
            self.node_fts, self.edge_fts, self.graph_fts, self.hidden, self.adj_matrix
        )
        self.assertEqual(out_no_triplet.shape, (self.batch_size, self.nb_nodes, self.hidden_dim))
        self.assertIsNone(tri_msgs_no_triplet)
        
        # Test FullGAT (ignores adjacency matrix)
        full_gat = FullGAT(self.hidden_dim, self.hidden_dim, nb_triplet_fts=self.nb_triplet_fts)
        out_full, _ = full_gat(self.node_fts, self.edge_fts, self.graph_fts, self.hidden, self.adj_matrix)
        self.assertEqual(out_full.shape, (self.batch_size, self.nb_nodes, self.hidden_dim))
    
    def test_mpnn(self):
        """Test Message Passing Neural Networks."""
        model = MPNN(self.hidden_dim, self.hidden_dim, nb_triplet_fts=self.nb_triplet_fts)
        out, tri_msgs = model(self.node_fts, self.edge_fts, self.graph_fts, self.hidden, self.adj_matrix)
        
        # Check output shapes
        self.assertEqual(out.shape, (self.batch_size, self.nb_nodes, self.hidden_dim))
        self.assertEqual(tri_msgs.shape, (self.batch_size, self.nb_nodes, self.nb_nodes, self.hidden_dim))
    
    def test_deep_sets_layer(self):
        """Test DeepSetsLayer."""
        model = DeepSetsLayer(self.hidden_dim, self.hidden_dim)
        out = model(self.node_fts)
        
        # Check output shape
        self.assertEqual(out.shape, (self.batch_size, self.nb_nodes, self.hidden_dim))
    
    def test_spec_former_conv(self):
        """Test SpecFormerConv."""
        model = SpecFormerConv(self.hidden_dim)
        # Create bases for testing
        bases = torch.randn(self.batch_size, self.nb_nodes, self.nb_nodes, self.hidden_dim)
        node_fts, edge_fts = model(self.node_fts, self.edge_fts, bases)
        
        # Check output shapes
        self.assertEqual(node_fts.shape, (self.batch_size, self.nb_nodes, self.hidden_dim))
        self.assertEqual(edge_fts.shape, (self.batch_size, self.nb_nodes, self.nb_nodes, self.hidden_dim))
    
    def test_spectralmpnn(self):
        """Test SpectralMPNN."""
        # Test with message passing
        model = SpectralMPNN(self.hidden_dim, self.hidden_dim, message_passing=True)
        out, edge_fts = model(self.node_fts, self.edge_fts, self.graph_fts, self.hidden, self.adj_matrix)
        
        # Check output shapes
        self.assertEqual(out.shape, (self.batch_size, self.nb_nodes, self.hidden_dim))
        self.assertEqual(edge_fts.shape, (self.batch_size, self.nb_nodes, self.nb_nodes, self.hidden_dim))

        # Test without message passing
        model_no_mp = SpectralMPNN(self.hidden_dim, self.hidden_dim, message_passing=False)
        out_no_mp, edge_fts_no_mp = model_no_mp(self.node_fts, self.edge_fts, self.graph_fts, self.hidden, self.adj_matrix)
        
        # Check output shapes remain the same without message passing
        self.assertEqual(out_no_mp.shape, (self.batch_size, self.nb_nodes, self.hidden_dim))
        self.assertEqual(edge_fts_no_mp.shape, (self.batch_size, self.nb_nodes, self.nb_nodes, self.hidden_dim))

        # Test processor with no edge feature.
        model_no_triplet = SpectralMPNN(self.hidden_dim, self.hidden_dim, nb_triplet_fts=None)
        out, edge_fts = model_no_triplet(self.node_fts, self.edge_fts, self.graph_fts, self.hidden, self.adj_matrix)
        
        # Check edge features are none when nb_triplet_fts = None
        self.assertEqual(out.shape, (self.batch_size, self.nb_nodes, self.hidden_dim))
        self.assertIsNone(edge_fts)

    def test_chebyshev_graph_conv(self):
        """Test PolynomialSpectralMPNN."""
        # Test with message passing
        model = PolynomialSpectralMPNN(self.hidden_dim, self.hidden_dim, K=3, message_passing=True)
        out, edge_fts = model(self.node_fts, self.edge_fts, self.graph_fts, self.hidden, self.adj_matrix)
        
        # Check output shapes
        self.assertEqual(out.shape, (self.batch_size, self.nb_nodes, self.hidden_dim))
        self.assertEqual(edge_fts.shape, (self.batch_size, self.nb_nodes, self.nb_nodes, self.hidden_dim))

        # Test without message passing
        model_no_mp = PolynomialSpectralMPNN(self.hidden_dim, self.hidden_dim, K=3, message_passing=False)
        out_no_mp, edge_fts_no_mp = model_no_mp(self.node_fts, self.edge_fts, self.graph_fts, self.hidden, self.adj_matrix)
        
        # Check output shapes remain the same without message passing
        self.assertEqual(out_no_mp.shape, (self.batch_size, self.nb_nodes, self.hidden_dim))
        self.assertEqual(edge_fts_no_mp.shape, (self.batch_size, self.nb_nodes, self.nb_nodes, self.hidden_dim))
        
        # Test with different K values
        model_k1 = PolynomialSpectralMPNN(self.hidden_dim, self.hidden_dim, K=1)
        out_k1, _ = model_k1(self.node_fts, self.edge_fts, self.graph_fts, self.hidden, self.adj_matrix)
        self.assertEqual(out_k1.shape, (self.batch_size, self.nb_nodes, self.hidden_dim))
        
        model_k5 = PolynomialSpectralMPNN(self.hidden_dim, self.hidden_dim, K=5)
        out_k5, _ = model_k5(self.node_fts, self.edge_fts, self.graph_fts, self.hidden, self.adj_matrix)
        self.assertEqual(out_k5.shape, (self.batch_size, self.nb_nodes, self.hidden_dim))

        # Test processor with no edge feature.
        model_no_triplet = PolynomialSpectralMPNN(self.hidden_dim, self.hidden_dim, nb_triplet_fts=None)
        out, edge_fts = model_no_triplet(self.node_fts, self.edge_fts, self.graph_fts, self.hidden, self.adj_matrix)
        
        # Check edge features are none when nb_triplet_fts = None
        self.assertEqual(out.shape, (self.batch_size, self.nb_nodes, self.hidden_dim))
        self.assertIsNone(edge_fts)

if __name__ == '__main__':
    unittest.main()
