import torch
from algo_reasoning.src.data import CLRSDataset
from algo_reasoning.src.specs import SPECS, Type

from algo_reasoning.src.algorithms.sorting import insertion_sort, bubble_sort, heapsort, quicksort
from algo_reasoning.src.algorithms.greedy import activity_selector, task_scheduling
from algo_reasoning.src.algorithms.dynamic_programming import matrix_chain_order, lcs_length, optimal_bst
from algo_reasoning.src.algorithms.searching import minimum, binary_search, quickselect
from algo_reasoning.src.algorithms.divide_and_conquer import find_maximum_subarray_kadane
from algo_reasoning.src.algorithms.strings import naive_string_matcher, kmp_matcher

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
    "optimal_bst": optimal_bst,
    "minimum": minimum,
    "binary_search": binary_search,
    "quickselect": quickselect,
    "find_maximum_subarray_kadane": find_maximum_subarray_kadane,
    "naive_string_matcher": naive_string_matcher, 
    "kmp_matcher": kmp_matcher
}

class CLRS30Test(unittest.TestCase):

    def compare_output(self, out, sample, algo, ignore_keys=[]):
        for k, hints in out.hints.items():
            if k in ignore_keys:
                continue

            out_len = hints.size(0)

            _, _, _type = SPECS[algo][k]

            if _type == Type.SCALAR:
                self.assertTrue(
                    torch.all(torch.isclose(hints, sample.hints[k][:out_len], rtol=1e-5, atol=1e-6)).item(),
                    msg=f"Key: {k}, Algo: {algo}, Hints: {hints}, Sample: {sample.hints[k][:out_len]}")
            else:
                self.assertTrue(torch.all(hints == sample.hints[k][:out_len]).item(),
                    msg=f"Key: {k}, Algo: {algo}, Hints: {hints}, Sample: {sample.hints[k][:out_len]}")

        for k, outputs in out.outputs.items():
            if k in ignore_keys:
                continue

            _, _, _type = SPECS[algo][k]

            if _type == Type.SCALAR:
                self.assertTrue(
                    torch.all(torch.isclose(outputs, sample.outputs[k][:out_len], rtol=1e-5, atol=1e-6)).item(),
                    msg=f"Key: {k}, Algo: {algo}, Outputs: {outputs}, Sample: {sample.outputs[k]}"
                )
            else:
                self.assertTrue(
                    torch.all(outputs == sample.outputs[k]).item(),
                    msg=f"Key: {k}, Algo: {algo}, Outputs: {outputs}, Sample: {sample.outputs[k]}"
                )
        

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

    def test_dynamic_programming(self):
        algorithms = ["matrix_chain_order", "lcs_length", "optimal_bst"]
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
            elif algo == "lcs_length":
                strings = torch.argmax(inp["key"].clone(), dim=1)
                strings_id = inp["string"].clone()
                inp["x"] = strings[strings_id == 0]
                inp["y"] = strings[strings_id == 1]

                del inp["key"]
                del inp["string"]

            out = algo_fn[algo](**inp, nb_nodes=nb_nodes)

            self.compare_output(out, sample, algo)

    def test_searching(self):
        algorithms = ["minimum", "binary_search", "quickselect"]
        ds = CLRSDataset(algorithms, "train", "tmp/CLRS30")

        for i in range(len(ds)):
            sample = ds[i]
            sample.squeeze(inplace=True)

            inp = sample.inputs.clone().to_dict()
            
            nb_nodes = inp["pos"].size(0)
            algo = sample.algorithm


            del inp["pos"]
            if algo == "binary_search":
                inp["x"] = inp["target"].item()
                inp["A"] = inp["key"].clone()

                del inp["key"]
                del inp["target"]
            else:
                inp["A"] = inp["key"].clone()

                del inp["key"]

            out = algo_fn[algo](**inp, nb_nodes=nb_nodes)

            self.compare_output(out, sample, algo, ignore_keys=["pivot"])

    def test_divide_and_conquer(self):
        algorithms = ["find_maximum_subarray_kadane"]
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

    def test_strings(self):
        algorithms = ["naive_string_matcher", "kmp_matcher"]

        ds = CLRSDataset(algorithms, "train", "tmp/CLRS30")

        for i in range(len(ds)):
            sample = ds[i]
            sample.squeeze(inplace=True)

            inp = sample.inputs.clone().to_dict()
            
            nb_nodes = inp["pos"].size(0)
            algo = sample.algorithm

            strings = torch.argmax(inp["key"].clone(), dim=1)
            strings_id = inp["string"].clone()
            inp["T"] = strings[strings_id == 0]
            inp["P"] = strings[strings_id == 1]

            del inp["pos"]
            del inp["key"]
            del inp["string"]

            out = algo_fn[algo](**inp, nb_nodes=nb_nodes)

            self.compare_output(out, sample, algo)

if __name__ == '__main__':
    unittest.main()




