import os
import numpy as np
import torch
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset, Sampler
from typing import List, Optional, Union

from algo_reasoning.src.specs import SPECS
from datasets import load_dataset

SPLITS = ["train", "val", "test"]
SAMPLERS = list(SPECS.keys())

class AlgorithmicData(Data):
    """A data object for CLRS data."""
    def __init__(self,
                pos_generator=None, 
                **kwargs):
        super().__init__(**kwargs)

        if pos_generator is not None:
            self.pos_generator = pos_generator

    def set_inputs(self, inputs, nb_nodes, inplace: bool = True, _strings_id: Optional[torch.Tensor] = None):
        """Set the inputs of the algorithm being executed."""
        data = self.clone() if not inplace else self

        data["inputs"] = AlgorithmicData()
        data["length"] = torch.tensor(0).float()

        for key, value in inputs.items():
            data["inputs"][key] = value.float()

        if _strings_id is None:
            data["inputs"]["pos"] = (torch.arange(nb_nodes) * 1.0) / nb_nodes
        else:
            len_str1 = torch.sum(_strings_id)
            len_str0 = nb_nodes - len_str1

            pos_str0 = torch.arange(len_str0) * 1.0 / len_str0
            pos_str1 = torch.arange(len_str1) * 1.0 / len_str1

            data["inputs"]["pos"] = torch.concatenate([pos_str0, pos_str1])

        if hasattr(data, 'pos_generator'):
            random_perm = torch.randperm(nb_nodes, generator=data.pos_generator)
            data["inputs"]["pos"] = data["inputs"]["pos"][random_perm]

            del data.pos_generator
        
        if not inplace:
            return data

    def set_outputs(self, outputs, inplace: bool = True):
        """Set the outputs of the algorithm being executed."""
        data = self.clone() if not inplace else self

        data["outputs"] = AlgorithmicData()
        data["max_length"] = data["length"].clone()

        for key, value in outputs.items():
            data["outputs"][key] = value.float()

        if not inplace:
            return data

    def increase_hints(self, hints, inplace: bool = True):
        """Set the hints of the algorithm being executed."""
        data = self.clone() if not inplace else self
        
        data["length"] += 1
        if "hints" not in data.keys():
            data["hints"] = AlgorithmicData()

            for key, value in hints.items():
                data["hints"][key] = value.float().unsqueeze(0)
        else:
            for key, value in hints.items():
                unsqueezed_value = value.float().unsqueeze(0)
                data["hints"][key] = torch.cat([data["hints"][key], unsqueezed_value], dim=0)
        
        if not inplace:
            return data

    def concat(self, other, inplace: bool = False):
        """Concatenate two AlgorithmicData objects."""
        data = self.clone() if not inplace else self

        for key, value in other.items():
            if key in data:
                if isinstance(value, AlgorithmicData):
                    data[key].concat(value)
                elif isinstance(value, str):
                    data[key] = value
                elif value.dim() > 0:
                    data[key] = torch.cat([data[key], value], dim=1)
                elif value.dim() == 0:
                    data[key] = torch.tensor([data[key], value], dtype=torch.float32)
                
            else:
                data[key] = value

        if not inplace:
            return data

    def unsqueeze(self, dim, inplace: bool = False):
        """Unsqueeze all data in AlgorithmicData objects."""
        data = self.clone() if not inplace else self

        for key, value in data.items():
            if isinstance(value, str) or isinstance(value, int) or isinstance(value, float):
                data[key] = value
            else:
                data[key] = value.unsqueeze(dim)

        if not inplace:
            return data
    
    def squeeze(self, dim: Optional[Union[int, List[int]]] = None, inplace: bool = False):
        """Squeeze all data in AlgorithmicData objects."""
        squeeze_fn = lambda x: x.squeeze(dim) if dim is not None else x.squeeze()

        data = self.clone() if not inplace else self

        for key, value in data.items():
            if isinstance(value, AlgorithmicData):
                data[key] = data[key].squeeze(dim)
            elif isinstance(value, str) or isinstance(value, int) or isinstance(value, float):
                data[key] = value
            else:
                data[key] = squeeze_fn(value)

        if not inplace:
            return data
    
    def to_dict(self):
        """Convert AlgorithmicData to a dictionary."""
        data_dict = dict()

        for key, value in self.items():
            if isinstance(value, AlgorithmicData):
                data_dict[key] = value.to_dict()
            else:
                data_dict[key] = value
        
        return data_dict
    
    def tolist(self):
        """Convert AlgorithmicData tensors to list."""
        data = self.clone()
        for key, value in self.items():
            if isinstance(value, AlgorithmicData):
                data[key] = value.tolist()
            elif isinstance(value, torch.Tensor):
                if value.dim() == 0:
                    data[key] = value.item()
                else:
                    data[key] = value.tolist()
            else:
                data[key] = value
        
        return data

class OriginalCLRSDataset(Dataset):
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

                ds = get_dataset(algorithm, self.split)

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

class AlgorithmicOutput(Data):
    def __init__(self, **kwargs):
        super(AlgorithmicOutput, self).__init__(**kwargs)

        assert "output" in kwargs, "output key must be provided to AlgorithmicOutput."
        assert "hidden_embeddings" in kwargs, "hidden_embeddings key must be provided to AlgorithmicOutput."

def idx_batched_data(idx: int, batched_data: AlgorithmicData) -> AlgorithmicData:
    """Get itens at idx for batched data."""
    inputs_dict = {k: v[idx] for k, v in batched_data.inputs.items()}
    inputs = AlgorithmicData(**inputs_dict)

    outputs_dict = {k: v[idx] for k, v in batched_data.outputs.items()}
    outputs = AlgorithmicData(**outputs_dict)

    hints_dict = {k: v[idx] for k, v in batched_data.hints.items()}
    hints = AlgorithmicData(**hints_dict)

    algorithm = batched_data.algorithm
    length = batched_data.length[idx]
    max_length = torch.max(length).long().item()

    return AlgorithmicData(
        algorithm=algorithm,
        inputs=inputs,
        outputs=outputs,
        hints=hints,
        max_length=max_length,
        length=length,
    )

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
    max_length = torch.max(hint_lengths).long()

    batched_hints = AlgorithmicData()
    aux_hint = hints[0]
    for k, v in aux_hint.items():
        new_shape = (len(hints), max_length) + v.shape[2:]

        batched_hints[k] = torch.zeros(*new_shape)
    
    for sample_idx, cur_sample in enumerate(hints):
        for k, v in cur_sample.items():
            cur_length = min(v.size(1), max_length)
            batched_hints[k][sample_idx:sample_idx+1, :cur_length] = v[:, :cur_length]
        
    return batched_hints, max_length

def collate(batch):
    """Collate a batch of data points."""
    for data in batch:
        assert isinstance(data, AlgorithmicData), f"Data must be of type AlgorithmicData, got {type(data)}."
        
        data.unsqueeze(0, inplace=True)

    batch = Batch.from_data_list(batch)

    batch.algorithm = batch[0].algorithm

    batch.inputs = Batch.from_data_list(batch.inputs)
    batch.outputs = Batch.from_data_list(batch.outputs)

    batched_hints, max_length =_batch_hints(batch.hints, batch.length)

    batch.hints = batched_hints
    batch.max_length = max_length
    return batch

def _preprocess(data_point, algorithm=None):
    """Convert sampled inputs into DataPoints."""
    inputs = AlgorithmicData()
    outputs = AlgorithmicData()
    hints = AlgorithmicData()

    length = torch.tensor(data_point['length'])
    max_length = length.clone()

    dict_inputs = data_point['inputs']
    dict_outputs = data_point['outputs']
    dict_hints = data_point['hints']

    for key, data in dict_inputs.items():
        inputs[key] = torch.tensor(data)

    for key, data in dict_outputs.items():
        outputs[key] = torch.tensor(data)

    for key, data in dict_hints.items():
        hints[key] = torch.tensor(data)
    
    return AlgorithmicData(inputs=inputs, hints=hints, length=length, outputs=outputs, max_length=max_length, algorithm=algorithm)


def get_dataset(algorithm, split):
    """Load the CLRS dataset from hugging face for the given algorithm or list of algorithms and split.
    
    Args:
        algorithm (str): The algorithm to get the dataset for.
        split (str): The split to get the dataset for.
    """
    if algorithm not in SAMPLERS:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Available algorithms are {list(SAMPLERS)}.")

    if split not in SPLITS:
        raise ValueError(f"Unknown split '{split}'. Available splits are {list(SPLITS)}.")
    
    # check if the dataset is already downloaded
    huggingface_dataset = load_dataset("ronaldalbrt/CLRS30", data_files=f"{algorithm}/{split}.json", split="train")

    return [_preprocess(dp, algorithm=algorithm) for dp in huggingface_dataset]