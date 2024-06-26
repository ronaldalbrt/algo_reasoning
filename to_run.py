from algo_reasoning.src.models.network import EncodeProcessDecode
from algo_reasoning.src.data import CLRSDataset, CLRSSampler, collate
from algo_reasoning.src.losses.CLRSLoss import CLRSLoss
from algo_reasoning.src.specs import CLRS_30_ALGS, SPECS, Stage, OutputClass
from algo_reasoning.src.eval import eval_function
from torch.utils.data import DataLoader
import torch

algorithms = ["minimum", "task_scheduling", "dfs", "topological_sort"]
model = EncodeProcessDecode(algorithms)

ds = CLRSDataset(algorithms, "train", "tmp/CLRS30")
sampler = CLRSSampler(ds, algorithms, 32, replacement=False)

loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate)

for obj in loader:
    result = model(obj)

    eval = eval_function(result, obj)
# datasets = []

# for algorithm in CLRS_30_ALGS:
#     datasets.append(load_dataset(algorithm, "train", "tmp/CLRS30"))
#     datasets.append(load_dataset(algorithm, "val", "tmp/CLRS30"))
#     datasets.append(load_dataset(algorithm, "test", "tmp/CLRS30"))

# algorithms = CLRS_30_ALGS

# ds = CLRSDataset(algorithms, "train", "tmp/CLRS30")
# sampler = CLRSSampler(ds, algorithms, 32, replacement=False)

# loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate)

# for obj in loader:
#     print(obj.algorithm)