import unittest
import torch
import torch.nn as nn
from algo_reasoning.src.models.network import EncodeProcessDecode, build_processor
from algo_reasoning.src.data import AlgorithmicData
from algo_reasoning.src.sampler import CLRSDataset
from algo_reasoning.src.specs import CLRS_30_ALGS
from utils import _get_algorithm_outputs

class NetworkTest(unittest.TestCase):
    def setUp(self):
        # Common test parameters
        self.batch_size = 2
        self.nb_nodes = 4
        self.hidden_dim = 8
        self.test_algorithms = CLRS_30_ALGS
        
        # Create a dataset for testing
        self.dataset = CLRSDataset(self.test_algorithms, self.nb_nodes, self.batch_size, 1000)
        
        # Define expected output keys for each algorithm
        self.expected_outputs = {algorithm: _get_algorithm_outputs(algorithm) for algorithm in self.test_algorithms}
    
    def test_build_processor(self):
        """Test the build_processor function."""
        processors = ["pgn", "mpnn", "gat", "fullgat", "spectralmpnn", "chebconv"]
        
        for proc_name in processors:
            processor = build_processor(proc_name, self.hidden_dim, nb_triplet_fts=4)
            self.assertIsNotNone(processor, f"Failed to build processor: {proc_name}")

    def test_model_initialization(self):
        """Test initialization of the EncodeProcessDecode model."""
        model = EncodeProcessDecode(self.test_algorithms, hidden_dim=self.hidden_dim)

        self.assertIsInstance(model.processor, nn.Module)

        # Test encoders.
        self.assertIsInstance(model.encoders, nn.ModuleDict)
        self.assertEqual(len(model.encoders), len(self.test_algorithms))

        # Test decoders.
        self.assertIsInstance(model.decoders, nn.ModuleDict)
        self.assertEqual(len(model.encoders), len(self.test_algorithms))
        
        # Test with different parameters
        model_with_lstm = EncodeProcessDecode(self.test_algorithms, hidden_dim=self.hidden_dim, use_lstm=True)
        self.assertTrue(model_with_lstm.use_lstm)
        self.assertTrue(hasattr(model_with_lstm, 'lstm'))
        
        model_no_hints = EncodeProcessDecode(self.test_algorithms, hidden_dim=self.hidden_dim, encode_hints=False, decode_hints=False)
        self.assertFalse(any(encoder.encode_hints for encoder in model_no_hints.encoders.values()))
        
        model_hard_hints = EncodeProcessDecode(self.test_algorithms, hidden_dim=self.hidden_dim, soft_hints=False)
        self.assertFalse(any(encoder.soft_hint for encoder in model_hard_hints.encoders.values()))
        
        # Test with different processors
        for processor_name in ["pgn", "mpnn", "gat", "spectralmpnn"]:
            model_with_processor = EncodeProcessDecode(self.test_algorithms, hidden_dim=self.hidden_dim, processor=processor_name)
            self.assertIsNotNone(model_with_processor.processor)

    def test_one_step_prediction(self):
        """Test function for prediction of next step hints and outputs."""
        # Get a sample from the dataset
        sample = None
        while sample is None or not (sample.algorithm == self.test_algorithms[0]):
            sample = next(iter(self.dataset))
        
        if sample is None:
            self.fail(f"Could not find a sample for algorithm {self.test_algorithms[0]}")
        
        # Initialize model
        model = EncodeProcessDecode(self.test_algorithms, hidden_dim=self.hidden_dim)
        
        # Create hidden state with correct dimensions
        # Note: sample.inputs.pos has shape [batch_size, nb_nodes]
        nb_nodes = sample.inputs.pos.shape[1]
        hidden = torch.zeros(self.batch_size, nb_nodes, self.hidden_dim)
        
        # Test one step prediction
        output_pred, hidden, _ = model._one_step_prediction(sample, hidden)
        
        # Check outputs
        self.assertEqual(hidden.shape, (self.batch_size, nb_nodes, self.hidden_dim))
        self.assertIsInstance(output_pred, AlgorithmicData)
        self.assertIsInstance(output_pred.hints, AlgorithmicData)
        self.assertIsInstance(output_pred.outputs, AlgorithmicData)
        
        # Check that the expected output is present
        expected_key = self.expected_outputs[sample.algorithm]
        self.assertIn(expected_key, output_pred.outputs)
        
        # Test with different parameters
        model_hard_hints = EncodeProcessDecode(self.test_algorithms, hidden_dim=self.hidden_dim, soft_hints=False)
        output_pred, hidden, _ = model_hard_hints._one_step_prediction(sample, hidden)
        self.assertEqual(hidden.shape, (self.batch_size, nb_nodes, self.hidden_dim))
        self.assertIsInstance(output_pred, AlgorithmicData)
        
        model_no_triplet = EncodeProcessDecode(self.test_algorithms, hidden_dim=self.hidden_dim, nb_triplet_fts=None)
        output_pred, hidden, _ = model_no_triplet._one_step_prediction(sample, hidden)
        self.assertEqual(hidden.shape, (self.batch_size, nb_nodes, self.hidden_dim))
        self.assertIsInstance(output_pred, AlgorithmicData)

    def test_forward_pass(self):
        """Test the forward pass of the EncodeProcessDecode model."""
        # Get a sample from the dataset
        sample = None
        while sample is None or not (sample.algorithm == self.test_algorithms[0]):
            sample = next(iter(self.dataset))
        
        if sample is None:
            self.fail(f"Could not find a sample for algorithm {self.test_algorithms[0]}")
        
        # Initialize model
        model = EncodeProcessDecode(self.test_algorithms, hidden_dim=self.hidden_dim)
        
        try:
            # Forward pass
            output = model(sample)
            
            # Check outputs
            self.assertIn("output", output)
            self.assertIn("hidden_embeddings", output)
            
            # Check that the expected output is present
            expected_key = self.expected_outputs[sample.algorithm]
            self.assertIn(expected_key, output.output.outputs)
            
            # Check hidden embeddings shape
            nb_nodes = sample.inputs.pos.shape[1]
            self.assertEqual(output.hidden_embeddings.shape[2], nb_nodes)
            self.assertEqual(output.hidden_embeddings.shape[3], self.hidden_dim)
        except Exception as e:
            self.fail(f"Forward pass failed: {str(e)}")
    
    def test_model_with_different_parameters(self):
        """Test the model with different parameter combinations."""
        # Get a sample from the dataset
        sample = None
        while sample is None or not (sample.algorithm == self.test_algorithms[0]):
            sample = next(iter(self.dataset))
        
        if sample is None:
            self.fail(f"Could not find a sample for algorithm {self.test_algorithms[0]}")
        
        try:
            # Test with encode_hints=False, decode_hints=False
            model_no_hints = EncodeProcessDecode(
                self.test_algorithms, 
                hidden_dim=self.hidden_dim, 
                encode_hints=False, 
                decode_hints=False
            )
            output = model_no_hints(sample)
            self.assertEqual(len(output.output.hints), 0)
            
            # Test with soft_hints=False
            model_hard_hints = EncodeProcessDecode(
                self.test_algorithms, 
                hidden_dim=self.hidden_dim, 
                soft_hints=False
            )
            output = model_hard_hints(sample)
            self.assertIn("output", output)
            
            # Test with different processors
            for processor_name in ["pgn", "mpnn"]:
                model_with_processor = EncodeProcessDecode(
                    self.test_algorithms, 
                    hidden_dim=self.hidden_dim, 
                    processor=processor_name
                )
                output = model_with_processor(sample)
                self.assertIn("output", output)
                
            # Test with LSTM
            model_with_lstm = EncodeProcessDecode(
                self.test_algorithms, 
                hidden_dim=self.hidden_dim, 
                use_lstm=True
            )
            output = model_with_lstm(sample)
            self.assertIn("output", output)
            
            # Test with dropout
            model_with_dropout = EncodeProcessDecode(
                self.test_algorithms, 
                hidden_dim=self.hidden_dim, 
                dropout_prob=0.1
            )
            output = model_with_dropout(sample)
            self.assertIn("output", output)
        except Exception as e:
            self.fail(f"Model with different parameters failed: {str(e)}")
    
    def test_model_across_algorithms(self):
        """Test the model across different algorithms."""
        model = EncodeProcessDecode(self.test_algorithms, hidden_dim=self.hidden_dim)
        
        for algorithm in self.test_algorithms:
            # Get a sample for this algorithm
            sample = None
            while sample is None or not (sample.algorithm == algorithm):
                sample = next(iter(self.dataset))

            if sample is None:
                self.fail(f"Could not find a sample for algorithm {algorithm}")
            
            try:
                # Forward pass
                output = model(sample)
                
                # Check outputs
                self.assertIn("output", output)
                self.assertIn("hidden_embeddings", output)
                
                # Check that the expected output is present
                expected_key = self.expected_outputs[algorithm]
                self.assertIn(expected_key, output.output.outputs, 
                             f"Expected output key {expected_key} not found for algorithm {algorithm}")
            except Exception as e:
                self.fail(f"Model failed for algorithm {algorithm}: {str(e)}")


if __name__ == '__main__':
    unittest.main()
