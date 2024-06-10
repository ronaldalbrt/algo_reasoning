""" This file contains custom data classes for CLRS data. It mainly concerns with with sparseing the graphs and converting them into pytorch geometric data objects.
Further it provides a custom dataloader that automatically pads hints to the maximum length of the batch. This is necessary because the hints are not padded in the dataset."""


import clrs
import numpy as np
import numpy as np
from torch_geometric.data import Data, Batch
import tensorflow_datasets as tfds
from loguru import logger
import torch

SPLITS = ["train", "val", "test"]

SAMPLERS = [
    'insertion_sort',
    'bubble_sort',
    'heapsort',
    'quicksort',
    'quickselect',
    'minimum',
    'binary_search',
    'find_maximum_subarray',
    'find_maximum_subarray_kadane',
    'matrix_chain_order',
    'lcs_length',
    'optimal_bst',
    'activity_selector',
    'task_scheduling',
    'dfs',
    'topological_sort',
    'strongly_connected_components',
    'articulation_points',
    'bridges',
    'bfs',
    'mst_kruskal',
    'mst_prim',
    'bellman_ford',
    'dag_shortest_paths',
    'dijkstra',
    'floyd_warshall',
    'bipartite_matching',
    'naive_string_matcher',
    'kmp_matcher',
    'segments_intersect',
    'graham_scan',
    'jarvis_march',
]

# Loader Example:
# from torch.utils.data import DataLoader
# loader = DataLoader(ds, 32, collate_fn=collate)

def to_torch(value):
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value).to(torch.float32)
    elif isinstance(value, torch.Tensor):
        return value
    else:
        return torch.tensor(value, dtype=torch.float32)

def _preprocess(data_point, algorithm=None):
    """Convert sampled inputs into DataPoints."""
    inputs = CLRSData()
    outputs = CLRSData()
    hints = CLRSData()
    length = None

    for name, data in data_point.items():
        if name == 'lengths':
            length = to_torch(np.copy(data))
            continue
        data_point_name = name.split('_')
        name = '_'.join(data_point_name[1:])
        stage = data_point_name[0]

        if stage == "input":
            inputs[name] = to_torch(np.copy(data)).unsqueeze(0)
        elif stage == "output":
            outputs[name] = to_torch(np.copy(data)).unsqueeze(0)
        else:
            hints[name] = to_torch(np.copy(data)).unsqueeze(0)
    
    return CLRSData(inputs=inputs, hints=hints, length=length, outputs=outputs, algorithm=algorithm)

def collate(batch):
    """Collate a batch of data points."""
    batch = Batch.from_data_list(batch)

    batch.algorithm = batch[0].algorithm

    batch.inputs = Batch.from_data_list(batch.inputs)
    batch.hints = Batch.from_data_list(batch.hints)
    batch.outputs = Batch.from_data_list(batch.outputs)

    return batch

def load_dataset(algorithm, split, local_dir):
    """Load the CLRS dataset for the given algorithm or list of algorithms and split.
    
    Args:
        algorithm (str): The algorithm to get the dataset for.
        split (str): The split to get the dataset for.
        local_dir (str): The directory to download the dataset to.
    """
    if algorithm not in SAMPLERS:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Available algorithms are {list(SAMPLERS.keys())}.")

    if split not in SPLITS:
        raise ValueError(f"Unknown split '{split}'. Available splits are {list(SPLITS)}.")
    
    # check if the dataset is already downloaded
    try:
        dataset = tfds.load(f'clrs_dataset/{algorithm}_{split}', data_dir=local_dir, split=split, download=False)
    except:
        logger.info(f"Downloading dataset for algorithm '{algorithm}'...")
        clrs.create_dataset(folder=local_dir, algorithm=algorithm, split=split, batch_size=32)
        dataset = tfds.load(f'clrs_dataset/{algorithm}_{split}', data_dir=local_dir, split=split, download=False)

    dataset_it = dataset.as_numpy_iterator()

    return [_preprocess(i, algorithm=algorithm) for i in dataset_it]

class CLRSData(Data):
    """A data object for CLRS data."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def concat(self, other):
        """Concatenate two CLRSData objects."""
        for key, value in other.items():
            if key in self:
                if isinstance(value, CLRSData):
                    self[key].concat(value)
                elif isinstance(value, str):
                    self[key] = value
                elif value.dim() > 0:
                    self[key] = torch.cat([self[key], value], dim=1)
                elif value.dim() == 0:
                    self[key] = torch.tensor([self[key], value], dtype=torch.float32)
                
            else:
                self[key] = value


