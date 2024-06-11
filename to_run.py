from algo_reasoning.src.models.network import EncodeProcessDecode
from algo_reasoning.src.data.data import load_dataset, collate
from algo_reasoning.src.losses.CLRSLoss import CLRSLoss
from algo_reasoning.src.data.specs import CLRS_30_ALGS
from torch.utils.data import DataLoader

model = EncodeProcessDecode(["articulation_points"])

articulation_points_train = load_dataset("articulation_points", "train", "tmp/CLRS30")

loader = DataLoader(articulation_points_train, 32, collate_fn=collate)

obj = next(iter(loader))

result = model(obj)

loss = CLRSLoss("l2")

result, result_hint = loss(result, obj)

result.backward()

print([p.grad for p in model.parameters()])
# datasets = []

# for algorithm in CLRS_30_ALGS:
#     datasets.append(load_dataset(algorithm, "train", "tmp/CLRS30"))
#     datasets.append(load_dataset(algorithm, "val", "tmp/CLRS30"))
#     datasets.append(load_dataset(algorithm, "test", "tmp/CLRS30"))