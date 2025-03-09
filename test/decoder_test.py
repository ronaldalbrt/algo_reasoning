import unittest
import torch
from algo_reasoning.src.models.decoder import Decoder, NodeBaseDecoder, NodePointerDecoder, EdgeBaseDecoder, EdgePointerDecoder, GraphBaseDecoder, GraphPointerDecoder
from algo_reasoning.src.data import AlgorithmicData
from algo_reasoning.src.sampler import CLRSDataset
from algo_reasoning.src.specs import CLRS_30_ALGS
from utils import _get_algorithm_outputs


class DecoderTest(unittest.TestCase):
    def setUp(self):
        # Common test parameters
        self.batch_size = 2
        self.nb_nodes = 4
        self.hidden_dim = 8
        self.edge_dim = 16
        self.graph_dim = 8
        
        # Create sample node, edge, and graph features for basic decoder tests
        self.node_fts = torch.randn(self.batch_size, self.nb_nodes, 3 * self.hidden_dim)
        self.edge_fts = torch.randn(self.batch_size, self.nb_nodes, self.nb_nodes, self.edge_dim)
        self.graph_fts = torch.randn(self.batch_size, self.graph_dim)
        
        # Create a dataset for testing
        self.test_algorithms = CLRS_30_ALGS
        self.dataset = CLRSDataset(self.test_algorithms, self.nb_nodes, self.batch_size, 1000)
    
    def test_node_decoders(self):
        """Test node decoders."""
        # Test NodeBaseDecoder
        node_base_decoder = NodeBaseDecoder(1, 3 * self.hidden_dim, self.edge_dim, self.graph_dim)
        output = node_base_decoder(self.node_fts, self.edge_fts, self.graph_fts)
        self.assertEqual(output.shape, (self.batch_size, self.nb_nodes))
        
        # Test NodePointerDecoder with correct dimensions
        node_pointer_decoder = NodePointerDecoder(1, 3 * self.hidden_dim, self.edge_dim, self.graph_dim)
        output = node_pointer_decoder(self.node_fts, self.edge_fts, self.graph_fts)
        self.assertEqual(output.shape, (self.batch_size, self.nb_nodes, self.nb_nodes))
    
    def test_edge_decoders(self):
        """Test edge decoders."""
        # Test EdgeBaseDecoder
        edge_base_decoder = EdgeBaseDecoder(1, 3 * self.hidden_dim, self.edge_dim, self.graph_dim)
        output = edge_base_decoder(self.node_fts, self.edge_fts, self.graph_fts)
        self.assertEqual(output.shape, (self.batch_size, self.nb_nodes, self.nb_nodes))
        
        # Test EdgePointerDecoder with correct dimensions
        edge_pointer_decoder = EdgePointerDecoder(1, 3 * self.hidden_dim, self.edge_dim, self.graph_dim)
        output = edge_pointer_decoder(self.node_fts, self.edge_fts, self.graph_fts)
        self.assertEqual(output.shape, (self.batch_size, self.nb_nodes, self.nb_nodes, self.nb_nodes))
    
    def test_graph_decoders(self):
        """Test graph decoders."""
        # Test GraphBaseDecoder
        graph_base_decoder = GraphBaseDecoder(1, 3 * self.hidden_dim, self.edge_dim, self.graph_dim)
        output = graph_base_decoder(self.node_fts, self.edge_fts, self.graph_fts)
        self.assertEqual(output.shape, (self.batch_size,))
        
        # Skip GraphPointerDecoder test for now as it requires special handling
    
    def test_decoder_initialization(self):
        """Test decoder initialization for all algorithms."""
        for algorithm in self.test_algorithms:
            # Test with default parameters
            decoder = Decoder(algorithm, 3 * self.hidden_dim, self.edge_dim, self.graph_dim)
            self.assertEqual(decoder.hidden_dim, 3 * self.hidden_dim)
            self.assertTrue(decoder.decode_hints)
            
            # Test with decode_hints=False
            decoder = Decoder(algorithm, 3 * self.hidden_dim, self.edge_dim, self.graph_dim, decode_hints=False)
            self.assertFalse(decoder.decode_hints)
    
    def test_decoder_forward_with_different_params(self):
        """Test decoder forward pass with different parameter combinations."""
        algorithm = "insertion_sort"
        
        # Get a sample from the dataset
        sample = None
        while sample is None or not (sample.algorithm == algorithm):
            sample = next(iter(self.dataset))
                
        # Create node, edge, and graph features with correct dimensions
        nb_nodes = sample.inputs.pos.shape[0]
        # Add batch dimension
        node_fts = torch.randn(1, nb_nodes, 3 * self.hidden_dim)
        edge_fts = torch.randn(1, nb_nodes, nb_nodes, self.edge_dim)
        graph_fts = torch.randn(1, self.graph_dim)
        
        # Test with default parameters (decode_hints=True)
        decoder = Decoder(algorithm, 3 * self.hidden_dim, self.edge_dim, self.graph_dim)
        
        try:
            output = decoder(node_fts, edge_fts, graph_fts)
            
            self.assertIsInstance(output, AlgorithmicData)
            self.assertEqual(output.algorithm, algorithm)
            
            # For insertion_sort, the output should have 'pred'
            self.assertIn("pred", output.outputs)
            
            # Check that at least one hint spec is present
            self.assertIn("pred_h", output.hints)
        except Exception as e:
            self.fail(f"Decoder with default parameters failed: {str(e)}")
        
        # Test with decode_hints=False
        decoder = Decoder(algorithm, 3 * self.hidden_dim, self.edge_dim, self.graph_dim, decode_hints=False)
        
        try:
            output = decoder(node_fts, edge_fts, graph_fts)
            
            self.assertIsInstance(output, AlgorithmicData)
            self.assertEqual(output.algorithm, algorithm)
            self.assertIn("pred", output.outputs)
            self.assertEqual(len(output.hints), 0)
        except Exception as e:
            self.fail(f"Decoder with decode_hints=False failed: {str(e)}")
    
    def test_decoder_across_algorithms(self):
        """Test decoder across a subset of algorithms."""
        # Define expected output keys for each algorithm


        expected_outputs = {algorithm: _get_algorithm_outputs(algorithm) for algorithm in self.test_algorithms}
        
        for algorithm in self.test_algorithms:
            # Get a sample from the dataset for this algorithm
            sample = None
            while sample is None or not (sample.algorithm == algorithm):
                sample = next(iter(self.dataset))
                    
            # Create node, edge, and graph features with correct dimensions
            nb_nodes = sample.inputs.pos.shape[0]
            # Add batch dimension
            node_fts = torch.randn(1, nb_nodes, 3 * self.hidden_dim)
            edge_fts = torch.randn(1, nb_nodes, nb_nodes, self.edge_dim)
            graph_fts = torch.randn(1, self.graph_dim)
            
            # Test with default parameters
            decoder = Decoder(algorithm, 3 * self.hidden_dim, self.edge_dim, self.graph_dim)
            
            try:
                output = decoder(node_fts, edge_fts, graph_fts)
                
                self.assertIsInstance(output, AlgorithmicData)
                self.assertEqual(output.algorithm, algorithm)
                
                # Check for the expected output key for this algorithm
                expected_key = expected_outputs[algorithm]
                self.assertIn(expected_key, output.outputs, f"Expected output key {expected_key} not found for algorithm {algorithm}")
                
                # If decode_hints is True, check that at least one hint spec is present
                if decoder.decode_hints:
                    # Each algorithm has different hint keys, but we can check that hints are not empty
                    self.assertGreater(len(output.hints), 0, f"No hints found for algorithm {algorithm}")
            except Exception as e:
                self.fail(f"Decoder failed for algorithm {algorithm}: {str(e)}")
    
    def test_decoder_with_categorical_outputs(self):
        """Test decoder with categorical outputs."""
        # Test with an algorithm that has categorical outputs
        algorithm = "lcs_length"
        
        # Create node, edge, and graph features with correct dimensions
        nb_nodes = 16  # Typical size for CLRS samples
        # Add batch dimension
        node_fts = torch.randn(1, nb_nodes, 3 * self.hidden_dim)
        edge_fts = torch.randn(1, nb_nodes, nb_nodes, self.edge_dim)
        graph_fts = torch.randn(1, self.graph_dim)
        
        # For categorical outputs, we need to ensure the decoder is properly initialized
        # with the correct number of categories
        decoder = Decoder(algorithm, 3 * self.hidden_dim, self.edge_dim, self.graph_dim)
        
        try:
            output = decoder(node_fts, edge_fts, graph_fts)
            
            self.assertIsInstance(output, AlgorithmicData)
            self.assertEqual(output.algorithm, algorithm)
            
            # Check that the categorical output 'b' is present
            self.assertIn("b", output.outputs)
        except Exception as e:
            self.fail(f"Decoder failed for algorithm {algorithm} with categorical outputs: {str(e)}")


if __name__ == '__main__':
    unittest.main()
