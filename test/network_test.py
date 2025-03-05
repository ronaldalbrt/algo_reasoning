import unittest
import torch
from algo_reasoning.src.models.network import EncodeProcessDecode, build_processor
from algo_reasoning.src.data import AlgorithmicData
from algo_reasoning.src.specs import SPECS, Type

class NetworkTest(unittest.TestCase):
    def setUp(self):
        # Common test parameters
        self.batch_size = 2
        self.nb_nodes = 4
        self.hidden_dim = 8
        self.algorithms = ["insertion_sort"]  # Use a simple algorithm for testing
        
        # Create a sample batch for testing
        self.batch = self._create_sample_batch()
    
    def _create_sample_batch(self):
        """Create a sample batch for testing."""
        # Create inputs
        inputs = AlgorithmicData()
        inputs["pos"] = torch.linspace(0, 1, self.nb_nodes).unsqueeze(0).repeat(self.batch_size, 1)
        inputs["key"] = torch.randn(self.batch_size, self.nb_nodes)
        
        # Create outputs
        outputs = AlgorithmicData()
        outputs["pred"] = torch.zeros(self.batch_size, self.nb_nodes, self.nb_nodes)
        for i in range(self.batch_size):
            outputs["pred"][i, torch.arange(self.nb_nodes), torch.randperm(self.nb_nodes)] = 1
        
        # Create hints
        hints = AlgorithmicData()
        hints["pred"] = torch.zeros(1, self.batch_size, self.nb_nodes, self.nb_nodes)
        for i in range(self.batch_size):
            hints["pred"][0, i, torch.arange(self.nb_nodes), torch.randperm(self.nb_nodes)] = 1
        
        # Create the batch
        batch = AlgorithmicData(
            algorithm=self.algorithms[0],
            inputs=inputs,
            outputs=outputs,
            hints=hints,
            length=torch.ones(self.batch_size),
            max_length=torch.tensor(1)
        )
        
        return batch
    
    def test_build_processor(self):
        """Test the build_processor function."""
        processors = ["pgn", "mpnn", "gat", "fullgat", "spectralmpnn", "chebconv"]
        
        for proc_name in processors:
            processor = build_processor(proc_name, self.hidden_dim, nb_triplet_fts=4)
            self.assertIsNotNone(processor, f"Failed to build processor: {proc_name}")

if __name__ == '__main__':
    unittest.main()
