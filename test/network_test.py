import unittest
import torch
import torch.nn as nn
from algo_reasoning.src.models.network import EncodeProcessDecode, build_processor
from algo_reasoning.src.data import AlgorithmicData
from algo_reasoning.src.sampler import CLRSDataset
from algo_reasoning.src.specs import SPECS, Type, CLRS_30_ALGS


class NetworkTest(unittest.TestCase):
    def setUp(self):
        # Common test parameters
        self.batch_size = 2
        self.nb_nodes = 4
        self.hidden_dim = 8
        self.algorithms = CLRS_30_ALGS  # Use a simple algorithm for testing
        self.train_steps = 2
        
        # Create a sample batch for testing
        self.batch = self._create_sample_batch()

    def _create_sample_batch(self):
        """Create a sample batch for testing."""
        train_dataset = CLRSDataset(self.algorithms, self.nb_nodes, self.batch_size, self.train_steps)

        batch = next(iter(train_dataset))

        return batch
        
    
    def test_build_processor(self):
        """Test the build_processor function."""
        processors = ["pgn", "mpnn", "gat", "fullgat", "spectralmpnn", "chebconv"]
        
        for proc_name in processors:
            processor = build_processor(proc_name, self.hidden_dim, nb_triplet_fts=4)
            self.assertIsNotNone(processor, f"Failed to build processor: {proc_name}")

    def test_intermediatemodels(self):
        """Test implementation of the EncodeProcessDecode model."""
        model = EncodeProcessDecode(self.algorithms, hidden_dim=self.hidden_dim)

        self.assertIsInstance(model.processor, nn.Module)

        # Test encoders.
        self.assertIsInstance(model.encoders, nn.ModuleDict)
        self.assertEqual(len(model.encoders), len(self.algorithms))

        # Test decoders.
        self.assertIsInstance(model.decoders, nn.ModuleDict)
        self.assertEqual(len(model.encoders), len(self.algorithms))

    def test_one_step_prediction(self):
        """Test function for prediction of next step hints and outputs."""
        batch = self.batch.clone()
        model = EncodeProcessDecode(self.algorithms, hidden_dim=self.hidden_dim)
        model_hardhints = EncodeProcessDecode(self.algorithms, hidden_dim=self.hidden_dim, soft_hints=False)
        model_nonetriplet = EncodeProcessDecode(self.algorithms, hidden_dim=self.hidden_dim, soft_hints=False, nb_triplet_fts=None)
        hidden = torch.zeros(self.batch_size, self.nb_nodes, self.hidden_dim)

        # Test output of vanilla one step prediction.
        output_pred, hidden, _ = model._one_step_prediction(batch, hidden)
        self.assertEqual(hidden.shape, (self.batch_size, self.nb_nodes, self.hidden_dim))
        
        self.assertIsInstance(output_pred, AlgorithmicData)
        self.assertIsInstance(output_pred.hints, AlgorithmicData)
        self.assertIsInstance(output_pred.outputs, AlgorithmicData)


        # Test output of one step prediction with hard hints.
        output_pred, hidden, _ = model_hardhints._one_step_prediction(batch, hidden)
        self.assertEqual(hidden.shape, (self.batch_size, self.nb_nodes, self.hidden_dim))
        
        self.assertIsInstance(output_pred, AlgorithmicData)
        self.assertIsInstance(output_pred.hints, AlgorithmicData)
        self.assertIsInstance(output_pred.outputs, AlgorithmicData)

        # Test output of one step prediction with no edge features.
        output_pred, hidden, _ = model_nonetriplet._one_step_prediction(batch, hidden)
        self.assertEqual(hidden.shape, (self.batch_size, self.nb_nodes, self.hidden_dim))
        
        self.assertIsInstance(output_pred, AlgorithmicData)
        self.assertIsInstance(output_pred.hints, AlgorithmicData)
        self.assertIsInstance(output_pred.outputs, AlgorithmicData)



if __name__ == '__main__':
    unittest.main()
