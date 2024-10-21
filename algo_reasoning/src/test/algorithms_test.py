import torch
from algo_reasoning.src.algorithms.sorting import insertion_sort, bubble_sort, heapsort, quicksort
from algo_reasoning.src.algorithms.greedy import activity_selector, task_scheduling
from algo_reasoning.src.algorithms.dynamic_programming import matrix_chain_order, lcs_length, optimal_bst
from algo_reasoning.src.data import CLRSDataset
from algo_reasoning.src.specs import SPECS, Type

import unittest

algo_fn = {
    "insertion_sort": insertion_sort,
    "bubble_sort": bubble_sort,
    "heapsort": heapsort,
    "quicksort": quicksort,
    "activity_selector": activity_selector,
    "task_scheduling": task_scheduling,
    "matrix_chain_order": matrix_chain_order,
    "lcs_length": lcs_length,
    "optimal_bst": optimal_bst

}

class CLRS30Test(unittest.TestCase):

    def compare_output(self, out, sample, algo):
        for k, hints in out.hints.items():
            out_len = hints.size(0)

            _, _, _type = SPECS[algo][k]

            if _type == Type.SCALAR:
                self.assertTrue(torch.all(torch.isclose(hints, sample.hints[k][:out_len])).item())
            else:
                self.assertTrue(torch.all(hints == sample.hints[k][:out_len]).item())

        for k, outputs in out.outputs.items():
            _, _, _type = SPECS[algo][k]

            if _type == Type.SCALAR:
                self.assertTrue(torch.all(torch.isclose(outputs, sample.outputs[k][:out_len])).item())
            else:
                self.assertTrue(torch.all(outputs == sample.outputs[k]).item())
        

    def test_sorting(self):
        #algorithms = ["insertion_sort", "bubble_sort", "heapsort", "quicksort"]
        algorithms = ["insertion_sort", "heapsort", "quicksort"]
        ds = CLRSDataset(algorithms, "train", "tmp/CLRS30")

        for i in range(len(ds)):
            sample = ds[i]
            sample.squeeze(inplace=True)

            inp = sample.inputs.clone().to_dict()
            
            nb_nodes = inp["pos"].size(0)
            algo = sample.algorithm
            
            inp["A"] = inp["key"].clone()
            del inp["pos"]
            del inp["key"]

            out = algo_fn[algo](**inp, nb_nodes=nb_nodes)

            self.compare_output(out, sample, algo)

    def test_greedy(self):
        algorithms = ["activity_selector", "task_scheduling"]
        ds = CLRSDataset(algorithms, "train", "tmp/CLRS30")

        for i in range(len(ds)):
            sample = ds[i]
            sample.squeeze(inplace=True)

            inp = sample.inputs.clone().to_dict()
            
            nb_nodes = inp["pos"].size(0)
            algo = sample.algorithm

            del inp["pos"]

            out = algo_fn[algo](**inp, nb_nodes=nb_nodes)

            self.compare_output(out, sample, algo)

    # TODO: Implement testing for LCS length
    def test_dynamic_programming(self):
        algorithms = ["matrix_chain_order", "optimal_bst"]
        ds = CLRSDataset(algorithms, "train", "tmp/CLRS30")

        for i in range(len(ds)):
            sample = ds[i]
            sample.squeeze(inplace=True)

            inp = sample.inputs.clone().to_dict()
            
            nb_nodes = inp["pos"].size(0)
            algo = sample.algorithm

            del inp["pos"]
            if algo == "optimal_bst":
                inp["p"] = inp["p"][:-1]

            out = algo_fn[algo](**inp, nb_nodes=nb_nodes)

            self.compare_output(out, sample, algo)

            
            
        


if __name__ == '__main__':
    unittest.main()




