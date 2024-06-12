from algo_reasoning.src.models.network import EncodeProcessDecode
from algo_reasoning.src.data import CLRSDataset, CLRSSampler, collate
from algo_reasoning.src.losses.CLRSLoss import CLRSLoss
from algo_reasoning.src.specs import CLRS_30_ALGS
from torch.utils.data import DataLoader

# model = EncodeProcessDecode(["articulation_points"])

# articulation_points_train = load_dataset("articulation_points", "train", "tmp/CLRS30")

# loader = DataLoader(articulation_points_train, 32, collate_fn=collate)

# for obj in loader:
#     result = model(obj)

#     loss = CLRSLoss(64)

#     loss = loss(result, obj)

#print([p.grad for p in model.parameters()])
# datasets = []

# for algorithm in CLRS_30_ALGS:
#     datasets.append(load_dataset(algorithm, "train", "tmp/CLRS30"))
#     datasets.append(load_dataset(algorithm, "val", "tmp/CLRS30"))
#     datasets.append(load_dataset(algorithm, "test", "tmp/CLRS30"))

algorithms = ["insertion_sort", "bfs", "quickselect"]

ds = CLRSDataset(algorithms, "train", "tmp/CLRS30")
sampler = CLRSSampler(ds, algorithms, 32)

loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate)

obj = next(iter(loader))

print(obj)