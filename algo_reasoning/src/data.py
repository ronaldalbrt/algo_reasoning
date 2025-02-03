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

import os
import numpy as np
import torch
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset, Sampler
from typing import List, Optional, Union, Dict, Tuple

from algo_reasoning.src.specs import SPECS
from datasets import load_dataset
from dotenv import load_dotenv

SPLITS = ["train", "val", "test"]
SAMPLERS = list(SPECS.keys())

class AlgorithmicData(Data):
    """
    --------------------------------------------------------------------------------------------------------------------------
    AlgorithmicData is a class that represents the data of an algorithmic task.
    --------------------------------------------------------------------------------------------------------------------------
    Args:
        pos_generator (torch.Generator, optional): A torch.Generator object to generate random permutations of the position tensor. Defaults to None.
    --------------------------------------------------------------------------------------------------------------------------
    """
    def __init__(self,
                pos_generator: Optional[torch.Generator] = None, 
                **kwargs):
        super().__init__(**kwargs)

        if pos_generator is not None:
            self.pos_generator = pos_generator

    def set_inputs(self, 
                   inputs: Dict[str, torch.Tensor],
                   nb_nodes: int, 
                   inplace: bool = True, 
                   _strings_id: Optional[torch.Tensor] = None) -> Optional['AlgorithmicData']:
        """
        --------------------------------------------------------------------------------------------------------------------------
        Set the inputs of the algorithm being executed.
        --------------------------------------------------------------------------------------------------------------------------
        Args:
            inputs (dict): The inputs of the algorithm.
            nb_nodes (int): The number of nodes in the graph.
            inplace (bool, optional): Whether to set the inputs inplace or not. Defaults to True.
            _strings_id (torch.Tensor, optional): A tensor containing the indices of the strings in the graph. Defaults to None.
        --------------------------------------------------------------------------------------------------------------------------
        """
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

    def set_outputs(self, 
                    outputs: Dict[str, torch.Tensor],
                    inplace: bool = True) -> Optional['AlgorithmicData']:
        """
        --------------------------------------------------------------------------------------------------------------------------
        Set the outputs of the algorithm being executed.
        --------------------------------------------------------------------------------------------------------------------------
        Args:
            outputs (dict): The outputs of the algorithm.
            inplace (bool, optional): Whether to set the outputs inplace or not. Defaults to True.
        --------------------------------------------------------------------------------------------------------------------------
        """
        data = self.clone() if not inplace else self

        data["outputs"] = AlgorithmicData()
        data["max_length"] = data["length"].clone()

        for key, value in outputs.items():
            data["outputs"][key] = value.float()

        if not inplace:
            return data

    def increase_hints(self, 
                       hints : Dict[str, torch.Tensor],
                       inplace: bool = True) -> Optional['AlgorithmicData']:
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

    def concat(self, 
               other: 'AlgorithmicData', 
               inplace: bool = False) -> Optional['AlgorithmicData']:
        """
        --------------------------------------------------------------------------------------------------------------------------
        Concatenate two AlgorithmicData objects.
        --------------------------------------------------------------------------------------------------------------------------
        Args:
            other (AlgorithmicData): The other AlgorithmicData object to concatenate.
            inplace (bool, optional): Whether to concatenate the objects inplace or not. Defaults to False.
        --------------------------------------------------------------------------------------------------------------------------
        """
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

    def unsqueeze(self, 
                dim: Optional[Union[int, List[int]]] = None,
                inplace: bool = False) -> Optional['AlgorithmicData']:
        """
        --------------------------------------------------------------------------------------------------------------------------
        Unsqueeze all data in AlgorithmicData objects.
        --------------------------------------------------------------------------------------------------------------------------
        Args:
            dim (int): The dimension to unsqueeze.
            inplace (bool, optional): Whether to unsqueeze the objects inplace or not. Defaults to False.
        --------------------------------------------------------------------------------------------------------------------------
        """
        data = self.clone() if not inplace else self

        for key, value in data.items():
            if isinstance(value, str) or isinstance(value, int) or isinstance(value, float):
                data[key] = value
            else:
                data[key] = value.unsqueeze(dim)

        if not inplace:
            return data
    
    def squeeze(self, 
                dim: Optional[Union[int, List[int]]] = None, 
                inplace: bool = False) -> Optional['AlgorithmicData']:
        """
        --------------------------------------------------------------------------------------------------------------------------
        Squeeze all data in AlgorithmicData objects.
        --------------------------------------------------------------------------------------------------------------------------
        Args:
            dim (int, optional): The dimension to squeeze. Defaults to None.
            inplace (bool, optional): Whether to squeeze the objects inplace or not. Defaults to False.
        --------------------------------------------------------------------------------------------------------------------------
        """
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
    
    def to_dict(self) -> Dict[str, torch.Tensor]:
        """
        --------------------------------------------------------------------------------------------------------------------------
        Convert AlgorithmicData to a dictionary.
        --------------------------------------------------------------------------------------------------------------------------
        Args:
            None
        --------------------------------------------------------------------------------------------------------------------------
        """
        data_dict = dict()

        for key, value in self.items():
            if isinstance(value, AlgorithmicData):
                data_dict[key] = value.to_dict()
            else:
                data_dict[key] = value
        
        return data_dict
    
    def tolist(self) -> 'AlgorithmicData':
        """
        --------------------------------------------------------------------------------------------------------------------------
        Convert tensors in AlgorithmicData to a list.
        --------------------------------------------------------------------------------------------------------------------------
        Args:
            None
        --------------------------------------------------------------------------------------------------------------------------
        """
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
    """
    --------------------------------------------------------------------------------------------------------------------------
    Dataset for the Original CLRSDataset, with this class execution samples are generated on the fly, and the original
    samples from https://github.com/google-deepmind/clrs are used
    --------------------------------------------------------------------------------------------------------------------------
    Args:
        algorithms (List[str]): The algorithms to sample from.
        split (str): The split to sample from.
        data_folder (str, optional): The folder to store the data. Defaults to "tmp/CLRS30".
    --------------------------------------------------------------------------------------------------------------------------
    """
    def __init__(self, 
                algorithms: List[str], 
                split: str, 
                data_folder: str = "tmp/CLRS30"):
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
    
    def __getitem__(self, idx: int):
        algorithm = None
        data_idx = 0

        for k, v in self.algo_start_idx.items():
            if idx >= v and idx < (v + self.n_datapoints[k]):
                algorithm = k
                data_idx = idx - v

                break

        return torch.load(f"{self.data_folder}/{algorithm}/{self.split}/{data_idx}", weights_only=False)

class CLRSSampler(Sampler[List[int]]):
    """
    --------------------------------------------------------------------------------------------------------------------------
    CLRSSampler for sampling from the Original Dataset.
    --------------------------------------------------------------------------------------------------------------------------
    Args:
        dataset (OriginalCLRSDataset): The dataset to sample from.
        algorithms (List[str]): The algorithms to sample from.
        batch_size (int): The batch size.
        replacement (bool, optional): Whether to sample with replacement or not. Defaults to False.
        seed (int, optional): The seed for the random generator. Defaults to None.
    --------------------------------------------------------------------------------------------------------------------------
    """
    def __init__(self, 
                dataset: OriginalCLRSDataset, 
                algorithms: List[str], 
                batch_size: int, 
                replacement: bool = False, 
                seed: Optional[int] = None):
        super().__init__()
        self.dataset = dataset
        self.algorithms = algorithms
        self.n_algorithms = len(self.algorithms)
        self.algo_start_idx = self.dataset.algo_start_idx
        self.seed = seed
        self._generator = torch.Generator().manual_seed(self.seed) if self.seed is not None else torch.Generator()
        
        self.replacement = replacement

        self.batch_size = batch_size

    def reset_generator(self, seed:int):
        self._generator.manual_seed(seed)
    
    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.replacement:
            for _ in range(len(self.dataset) // self.batch_size):
                algo_idx = torch.randint(0, self.n_algorithms, (1,), generator=self._generator).item()

                algorithm = self.algorithms[algo_idx]

                min_idx = self.dataset.algo_start_idx[algorithm]
                max_idx = min_idx + self.dataset.n_datapoints[algorithm]

                yield torch.randint(min_idx, max_idx, size=(self.batch_size,), dtype=torch.int64, generator=self._generator).tolist()

            if (len(self.dataset) % self.batch_size) != 0:
                yield torch.randint(min_idx, max_idx, size=(len(self.dataset) % self.batch_size,), dtype=torch.int64, generator=self._generator).tolist()
        else:
            n_samples = 1000 if self.dataset.split == "train" else 32
            n_batches_per_algo = (1000 + self.batch_size - 1) // self.batch_size if self.dataset.split == "train" else  (32 + self.batch_size - 1) // self.batch_size

            wo_replacement_algos = np.array([])
            idx_order = {alg: torch.randperm(n_samples, generator=self._generator) for alg in self.algorithms}

            for alg in self.algorithms:
                wo_replacement_algos = np.append(wo_replacement_algos, [alg]*n_batches_per_algo)

            wo_replacement_algos = wo_replacement_algos[torch.randperm(len(wo_replacement_algos), generator=self._generator).tolist()]

            curr_idx = {alg: 0 for alg in self.algorithms}
            for batch in wo_replacement_algos:
                curr_idx[batch] += 1
                idx_min = (curr_idx[batch] - 1) * self.batch_size
                idx_max = curr_idx[batch] * self.batch_size

                yield (self.algo_start_idx[batch] + idx_order[alg][idx_min:idx_max]).tolist()

class AlgorithmicOutput(Data):
    """
    --------------------------------------------------------------------------------------------------------------------------
    AlgorithmicOutput is a class that represents the output of an algorithmic task.
    --------------------------------------------------------------------------------------------------------------------------
    """
    def __init__(self, **kwargs):
        super(AlgorithmicOutput, self).__init__(**kwargs)

        assert "output" in kwargs, "output key must be provided to AlgorithmicOutput."
        assert "hidden_embeddings" in kwargs, "hidden_embeddings key must be provided to AlgorithmicOutput."

def idx_batched_data(idx: int, batched_data: AlgorithmicData) -> AlgorithmicData:
    """
    --------------------------------------------------------------------------------------------------------------------------
    Get itens at idx for batched data.
    --------------------------------------------------------------------------------------------------------------------------
    Args:
        idx (int): The index to get the data from.
        batched_data (AlgorithmicData): The batched data.
    --------------------------------------------------------------------------------------------------------------------------
    """
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

def _batch_hints(hints: List[AlgorithmicData], 
                hint_lengths: torch.Tensor) -> Tuple[AlgorithmicData, int]:
    """
    --------------------------------------------------------------------------------------------------------------------------
    Batches a trajectory of hints samples along the time axis per probe.

    Unlike i/o, hints have a variable-length time dimension. Before batching, each
    trajectory is padded to the maximum trajectory length.
    --------------------------------------------------------------------------------------------------------------------------
    Args:
        hints: A hint trajectory of `DataPoints`s indexed by time then probe
        hint_lengths: A tensor of shape (num probes,) containing the length of each trajectory.
    --------------------------------------------------------------------------------------------------------------------------
    Returns:
        A |num probes| list of `DataPoint`s with the time axis stacked into `data`, and a |sample| list containing the length of each trajectory.
    --------------------------------------------------------------------------------------------------------------------------
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

def collate(batch: List[AlgorithmicData]) -> Batch:
    """
    --------------------------------------------------------------------------------------------------------------------------
    Collate a batch of AlgorithmicData objects.
    --------------------------------------------------------------------------------------------------------------------------
    Args:
        batch (List[AlgorithmicData]): The batch of AlgorithmicData objects.
    --------------------------------------------------------------------------------------------------------------------------
    Returns:
        Batch: A single AlgorithmicData object with an aditional dimension with size equal to the len(batch).
    ----------------------------------------------------------------------------------------------------------------
    """
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

def _preprocess(data_point: Dict[str, torch.Tensor], 
                algorithm: str = None) -> AlgorithmicData:
    """
    --------------------------------------------------------------------------------------------------------------------------
    Convert the dictionaries from HuggingFace into AlgorithmicData objects.
    --------------------------------------------------------------------------------------------------------------------------
    Args:
        data_point (dict): The data point to convert.
        algorithm (str, optional): The algorithm to convert the data point to. Defaults to None.
    --------------------------------------------------------------------------------------------------------------------------
    Returns:
        AlgorithmicData: The converted data point.
    ----------------------------------------------------------------------------------------------------------------
    """
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


def get_dataset(
        algorithm: str, 
        split: str) -> List[AlgorithmicData]:
    """
    --------------------------------------------------------------------------------------------------------------------------
    Load the CLRS dataset from hugging face for the given algorithm or list of algorithms and split.
    --------------------------------------------------------------------------------------------------------------------------
    Args:
        algorithm (str): The algorithm to get the dataset for.
        split (str): The split to get the dataset for.
    --------------------------------------------------------------------------------------------------------------------------
    Returns:
        List[AlgorithmicData]: The list of AlgorithmicData objects.
    ----------------------------------------------------------------------------------------------------------------
    """
    if algorithm not in SAMPLERS:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Available algorithms are {list(SAMPLERS)}.")

    if split not in SPLITS:
        raise ValueError(f"Unknown split '{split}'. Available splits are {list(SPLITS)}.")
    
    load_dotenv()
    token = os.environ["HUGGINGFACE_TOKEN"] if "HUGGINGFACE_TOKEN" in os.environ else None
    huggingface_dataset = load_dataset("ronaldalbrt/CLRS30", data_files=f"{algorithm}/{split}.json", split="train", token=token)

    return [_preprocess(dp, algorithm=algorithm) for dp in huggingface_dataset]