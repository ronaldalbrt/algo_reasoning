import unittest
import torch
from algo_reasoning.src.models.encoder import Encoder, preprocess
from algo_reasoning.src.data import AlgorithmicData
from algo_reasoning.src.sampler import CLRSDataset
from algo_reasoning.src.specs import SPECS, Type, CLRS_30_ALGS


class EncoderTest(unittest.TestCase):
    def setUp(self):
        # Common test parameters
        self.batch_size = 2
        self.nb_nodes = 4
        self.hidden_dim = 8
        
        # Create a dataset for testing
        self.test_algorithms = CLRS_30_ALGS
        self.dataset = CLRSDataset(self.test_algorithms, self.nb_nodes, self.batch_size, 1000)
    
    def test_preprocess(self):
        """Test the preprocess function."""
        # Test scalar data
        scalar_data = torch.randn(self.batch_size, self.nb_nodes)
        processed_scalar = preprocess(scalar_data, Type.SCALAR, self.nb_nodes)
        self.assertEqual(processed_scalar.shape, (self.batch_size, self.nb_nodes, 1))
        
        # Test pointer data
        pointer_data = torch.randint(0, self.nb_nodes, (self.batch_size, self.nb_nodes))
        processed_pointer = preprocess(pointer_data, Type.POINTER, self.nb_nodes)
        self.assertEqual(processed_pointer.shape, (self.batch_size, self.nb_nodes, self.nb_nodes, 1))
        
        # Test categorical data
        categorical_data = torch.randn(self.batch_size, self.nb_nodes, 3)  # 3 categories
        processed_categorical = preprocess(categorical_data, Type.CATEGORICAL, self.nb_nodes)
        self.assertEqual(processed_categorical.shape, (self.batch_size, self.nb_nodes, 3))
    
    def test_encoder_initialization(self):
        """Test encoder initialization for all algorithms."""
        for algorithm in self.test_algorithms:
            # Test with default parameters
            encoder = Encoder(algorithm, hidden_dim=self.hidden_dim)
            self.assertEqual(encoder.hidden_dim, self.hidden_dim)
            self.assertEqual(encoder.algorithm, algorithm)
            self.assertTrue(encoder.encode_hints)
            self.assertTrue(encoder.soft_hint)
            
            # Test with encode_hints=False
            encoder = Encoder(algorithm, encode_hints=False, hidden_dim=self.hidden_dim)
            self.assertFalse(encoder.encode_hints)
            
            # Test with soft_hints=False
            encoder = Encoder(algorithm, soft_hints=False, hidden_dim=self.hidden_dim)
            self.assertFalse(encoder.soft_hint)
    
    def test_encoder_forward(self):
        """Test encoder forward pass."""
        # Test with a simple algorithm
        algorithm = "insertion_sort"
        
        # Get a sample from the dataset
        sample = next(iter(self.dataset))
        
        # Test with default parameters
        encoder = Encoder(algorithm, hidden_dim=self.hidden_dim)
        
        try:
            node_hidden, edge_hidden, graph_hidden, adj_mat = encoder(sample)
            
            # Check output shapes - note that the encoder adds a batch dimension
            nb_nodes = sample.inputs.pos.shape[1]
            self.assertEqual(node_hidden.shape[-2:], (nb_nodes, self.hidden_dim))
            self.assertEqual(edge_hidden.shape[-3:], (nb_nodes, nb_nodes, self.hidden_dim))
            self.assertEqual(graph_hidden.shape[-1:], (self.hidden_dim,))
            self.assertEqual(adj_mat.shape[-2:], (nb_nodes, nb_nodes))
        except Exception as e:
            self.fail(f"Encoder forward pass failed: {str(e)}")
    
    def test_encoder_with_different_params(self):
        """Test encoder with different parameter combinations."""
        algorithm = "insertion_sort"
        
        # Get a sample from the dataset
        sample = next(iter(self.dataset))
        
        # Test with encode_hints=False
        encoder = Encoder(algorithm, encode_hints=False, hidden_dim=self.hidden_dim)
        
        try:
            node_hidden, edge_hidden, graph_hidden, adj_mat = encoder(sample)
            
            # Check output shapes - note that the encoder adds a batch dimension
            nb_nodes = sample.inputs.pos.shape[1]
            self.assertEqual(node_hidden.shape[-2:], (nb_nodes, self.hidden_dim))
            self.assertEqual(edge_hidden.shape[-3:], (nb_nodes, nb_nodes, self.hidden_dim))
            self.assertEqual(graph_hidden.shape[-1:], (self.hidden_dim,))
            self.assertEqual(adj_mat.shape[-2:], (nb_nodes, nb_nodes))
        except Exception as e:
            self.fail(f"Encoder with encode_hints=False failed: {str(e)}")
        
        # Test with soft_hints=False
        encoder = Encoder(algorithm, soft_hints=False, hidden_dim=self.hidden_dim)
        
        try:
            node_hidden, edge_hidden, graph_hidden, adj_mat = encoder(sample)
            
            # Check output shapes - note that the encoder adds a batch dimension
            nb_nodes = sample.inputs.pos.shape[1]
            self.assertEqual(node_hidden.shape[-2:], (nb_nodes, self.hidden_dim))
            self.assertEqual(edge_hidden.shape[-3:], (nb_nodes, nb_nodes, self.hidden_dim))
            self.assertEqual(graph_hidden.shape[-1:], (self.hidden_dim,))
            self.assertEqual(adj_mat.shape[-2:], (nb_nodes, nb_nodes))
        except Exception as e:
            self.fail(f"Encoder with soft_hints=False failed: {str(e)}")
    
    def test_encoder_across_algorithms(self):
        """Test encoder across a subset of algorithms."""
        for algorithm in self.test_algorithms:
            # Get a sample from the dataset for this algorithm
            sample = None
            while sample is None or not (sample.algorithm == algorithm):
                sample = next(iter(self.dataset))
            
            # Test with default parameters
            encoder = Encoder(algorithm, hidden_dim=self.hidden_dim)
            
            try:
                node_hidden, edge_hidden, graph_hidden, adj_mat = encoder(sample)
                
                # Check output shapes - note that the encoder adds a batch dimension
                nb_nodes = sample.inputs.pos.shape[1]
                self.assertEqual(node_hidden.shape[-2:], (nb_nodes, self.hidden_dim))
                self.assertEqual(edge_hidden.shape[-3:], (nb_nodes, nb_nodes, self.hidden_dim))
                self.assertEqual(graph_hidden.shape[-1:], (self.hidden_dim,))
                self.assertEqual(adj_mat.shape[-2:], (nb_nodes, nb_nodes))
            except Exception as e:
                self.fail(f"Encoder failed for algorithm {algorithm}: {str(e)}")


if __name__ == '__main__':
    unittest.main()
