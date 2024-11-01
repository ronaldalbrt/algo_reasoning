import torch
from algo_reasoning.src.specs import CLRS_30_ALGS
from algo_reasoning.src.data import AlgorithmicData
from os import listdir

import json

if __name__ == '__main__':
    for alg in CLRS_30_ALGS:
        for split in ["train", "val", "test"]:
            curr_json = []
            cur_path = f"./tmp/CLRS30/{alg}/{split}/"
            files = [f for f in listdir(cur_path)]
            
            print(cur_path)
            for file in files:
                clrs_data = torch.load(cur_path + file)

                clrs_data_list = clrs_data.tolist()
                json_data = clrs_data_list.to_dict()
                curr_json.append(json_data)

            with open(f"./tmp/CLRS30/{alg}/{split}.json", "w") as f:
                json.dump(curr_json, f)
            
            
