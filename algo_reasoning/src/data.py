import os
import clrs
import numpy as np
import numpy as np
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset, Sampler
import tensorflow_datasets as tfds
from typing import List
import torch
from collections import OrderedDict

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
    'schedule'
]

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
    max_length = 0

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
            max_length = hints[name].shape[1]
    
    return CLRSData(inputs=inputs, hints=hints, length=length, outputs=outputs, max_length=torch.tensor(max_length).long(), algorithm=algorithm)

def _batch_hints(hints, hint_lengths):
    """Batches a trajectory of hints samples along the time axis per probe.

    Unlike i/o, hints have a variable-length time dimension. Before batching, each
    trajectory is padded to the maximum trajectory length.

    Args:
    hints: A hint trajectory of `DataPoints`s indexed by time then probe

    Returns:
    A |num probes| list of `DataPoint`s with the time axis stacked into `data`,
    and a |sample| list containing the length of each trajectory.
    """
    max_length = torch.max(hint_lengths).long().item()

    batched_hints = CLRSData()
    aux_hint = hints[0]
    for k, v in aux_hint.items():
        new_shape = (len(hints), max_length) + v.shape[2:]

        batched_hints[k] = torch.zeros(*new_shape)
    
    for sample_idx, cur_sample in enumerate(hints):
        for k, v in cur_sample.items():
            cur_length = v.size(1)
            batched_hints[k][sample_idx:sample_idx+1, :cur_length] = v

    return batched_hints, max_length

def collate(batch):
    """Collate a batch of data points."""
    for data in batch:
        assert isinstance(data, CLRSData), f"Data must be of type CLRSData, got {type(data)}."
        
        data.unsqueeze(0)

    batch = Batch.from_data_list(batch)

    batch.algorithm = batch[0].algorithm

    batch.inputs = Batch.from_data_list(batch.inputs)
    batch.outputs = Batch.from_data_list(batch.outputs)

    batched_hints, max_length =_batch_hints(batch.hints, batch.length)

    batch.hints = batched_hints
    batch.max_length = max_length
    return batch

def load_dataset(algorithm, split, local_dir):
    """Load the CLRS dataset for the given algorithm or list of algorithms and split.
    
    Args:
        algorithm (str): The algorithm to get the dataset for.
        split (str): The split to get the dataset for.
        local_dir (str): The directory to download the dataset to.
    """
    if algorithm not in SAMPLERS:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Available algorithms are {list(SAMPLERS)}.")

    if split not in SPLITS:
        raise ValueError(f"Unknown split '{split}'. Available splits are {list(SPLITS)}.")
    
    # check if the dataset is already downloaded
    try:
        dataset = tfds.load(f'clrs_dataset/{algorithm}_{split}', data_dir=local_dir, split=split, download=False)
    except:
        clrs.create_dataset(folder=local_dir, algorithm=algorithm, split=split, batch_size=32)
        dataset = tfds.load(f'clrs_dataset/{algorithm}_{split}', data_dir=local_dir, split=split, download=False)

    dataset_it = dataset.as_numpy_iterator()

    return [_preprocess(i, algorithm=algorithm) for i in dataset_it]

class CLRSData(Data):
    """A data object for CLRS data."""
    def __init__(self,
                pos_generator=None, 
                **kwargs):
        super().__init__(**kwargs)

        if pos_generator is not None:
            self.pos_generator = pos_generator

    def set_inputs(self, inputs, nb_nodes):
        """Set the inputs of the algorithm being executed."""
        data = self.clone()

        data["inputs"] = CLRSData()
        data["length"] = torch.tensor(0).float()

        for key, value in inputs.items():
            data["inputs"][key] = value.float()

        data["inputs"]["pos"] = ((torch.arange(nb_nodes) * 1.0)/nb_nodes).float()

        if hasattr(data, 'pos_generator'):
            random_perm = torch.randperm(nb_nodes, generator=data.pos_generator)
            data["inputs"]["pos"] = data["inputs"]["pos"][random_perm]

            del data.pos_generator
        
        return data

    def set_outputs(self, outputs):
        """Set the outputs of the algorithm being executed."""
        data = self.clone()

        data["outputs"] = CLRSData()
        data["max_length"] = data["length"].clone()

        for key, value in outputs.items():
            data["outputs"][key] = value.float()

        return data

    def increase_hints(self, hints):
        """Set the hints of the algorithm being executed."""
        data = self.clone()
        
        data["length"] += 1
        if "hints" not in data.keys():
            data["hints"] = CLRSData()

            for key, value in hints.items():
                data["hints"][key] = value.float().unsqueeze(0)
        else:
            for key, value in hints.items():
                unsqueezed_value = value.float().unsqueeze(0)
                data["hints"][key] = torch.cat([data["hints"][key], unsqueezed_value], dim=0)
        
        return data

    def concat(self, other):
        """Concatenate two CLRSData objects."""
        data = self.clone()

        for key, value in other.items():
            if key in data:
                if isinstance(value, CLRSData):
                    data[key].concat(value)
                elif isinstance(value, str):
                    data[key] = value
                elif value.dim() > 0:
                    data[key] = torch.cat([data[key], value], dim=1)
                elif value.dim() == 0:
                    data[key] = torch.tensor([data[key], value], dtype=torch.float32)
                
            else:
                data[key] = value

        return data

    def unsqueeze(self, dim):
        """Unsqueeze all data in CLRSData objects."""
        data = self.clone()

        for key, value in self.items():
            if isinstance(value, CLRSData):
                self[key].unsqueeze(dim)
            elif isinstance(value, str):
                self[key] = value
            else:
                self[key] = value.unsqueeze(dim)

        return data

class CLRSDataset(Dataset):
    def __init__(self, algorithms, split, data_folder="tmp/CLRS30"):
        self.algorithms = algorithms
        self.split = split
        self.data_folder = data_folder

        self.n_datapoints = {}

        self.algo_start_idx = {}
        self.curr_length = 0

        for algorithm in self.algorithms:
            if os.path.isdir(f"{self.data_folder}/{algorithm}/{self.split}"):
                self.n_datapoints[algorithm] = 1000 if split == "train" else 32
                continue
            else:
                if not os.path.isdir(f"{self.data_folder}/{algorithm}"):
                    os.mkdir(f"{self.data_folder}/{algorithm}")
                
                os.mkdir(f"{self.data_folder}/{algorithm}/{self.split}")

                ds = load_dataset(algorithm, self.split, self.data_folder)

                self.n_datapoints[algorithm] = 1000 if split == "train" else 32
                
                for i, obj in enumerate(ds):
                    torch.save(obj, f"{self.data_folder}/{algorithm}/{self.split}/{i}")

        for algorithm in self.algorithms: 
            self.algo_start_idx[algorithm] = self.curr_length
            self.curr_length += self.n_datapoints[algorithm]

    def __len__(self):
        return self.curr_length
    
    def __getitem__(self, idx):
        algorithm = None
        data_idx = 0

        for k, v in self.algo_start_idx.items():
            if idx >= v and idx < (v + self.n_datapoints[k]):
                algorithm = k
                data_idx = idx - v

                break

        return torch.load(f"{self.data_folder}/{algorithm}/{self.split}/{data_idx}", weights_only=False)

class CLRSSampler(Sampler[List[int]]):
    def __init__(self, dataset, algorithms, batch_size, replacement=False, generator=None):
        super().__init__()
        self.dataset = dataset
        self.algorithms = algorithms
        self.n_algorithms = len(self.algorithms)
        self.algo_start_idx = self.dataset.algo_start_idx
        self.generator = generator
        
        self.replacement = replacement

        self.batch_size = batch_size

        if generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            self.generator = torch.Generator()
            self.generator.manual_seed(seed)
        else:
            self.generator = generator

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.replacement:
            for _ in range(len(self.dataset) // self.batch_size):
                algo_idx = torch.randint(0, self.n_algorithms, (1,), generator=self.generator).item()

                algorithm = self.algorithms[algo_idx]

                min_idx = self.dataset.algo_start_idx[algorithm]
                max_idx = min_idx + self.dataset.n_datapoints[algorithm]

                yield torch.randint(min_idx, max_idx, size=(self.batch_size,), dtype=torch.int64, generator=self.generator).tolist()

            if (len(self.dataset) % self.batch_size) != 0:
                yield torch.randint(min_idx, max_idx, size=(len(self.dataset) % self.batch_size,), dtype=torch.int64, generator=self.generator).tolist()
        else:
            n_samples = 1000 if self.dataset.split == "train" else 32
            n_batches_per_algo = (1000 + self.batch_size - 1) // self.batch_size if self.dataset.split == "train" else  (32 + self.batch_size - 1) // self.batch_size

            wo_replacement_algos = np.array([])
            idx_order = {alg: torch.randperm(n_samples, generator=self.generator) for alg in self.algorithms}

            for alg in self.algorithms:
                wo_replacement_algos = np.append(wo_replacement_algos, [alg]*n_batches_per_algo)

            wo_replacement_algos = wo_replacement_algos[torch.randperm(len(wo_replacement_algos), generator=self.generator).tolist()]

            curr_idx = {alg: 0 for alg in self.algorithms}
            for batch in wo_replacement_algos:
                curr_idx[batch] += 1
                idx_min = (curr_idx[batch] - 1) * self.batch_size
                idx_max = curr_idx[batch] * self.batch_size

                yield (self.algo_start_idx[batch] + idx_order[alg][idx_min:idx_max]).tolist()

class CLRSOutput(OrderedDict):
    def __init__(self, **kwargs):
        super(CLRSOutput, self).__init__(**kwargs)

        assert "output" in kwargs, "output key must be provided to CLRSOutput."
        assert "hidden_embeddings" in kwargs, "hidden_embeddings key must be provided to CLRSOutput."