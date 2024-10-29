import torch
from algo_reasoning.src.specs import CLRS_30_ALGS
from algo_reasoning.src.data import AlgorithmicData
from os import listdir

for alg in CLRS_30_ALGS:
    for split in ["train", "val", "test"]:
        cur_path = f"./tmp/CLRS30/{alg}/{split}/"
        files = [f for f in listdir(cur_path)]
        print(cur_path)
        for file in files:
            clrs_data = torch.load(cur_path + file)

            inputs = AlgorithmicData(**clrs_data.inputs.to_dict()).squeeze()
            hints = AlgorithmicData(**clrs_data.hints.to_dict()).squeeze()
            outputs =  AlgorithmicData(**clrs_data.outputs.to_dict()).squeeze()
            algorithm = clrs_data.algorithm
            length = torch.tensor(clrs_data.length)
            max_length = torch.tensor(clrs_data.max_length)

            new_data = AlgorithmicData(inputs=inputs, hints=hints, outputs=outputs, algorithm=algorithm, length=length, max_length=max_length)

            torch.save(new_data, cur_path + file)
