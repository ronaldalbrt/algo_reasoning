import torch
from algo_reasoning.src.data import OriginalCLRSDataset
from algo_reasoning.src.specs import SPECS, Type

from algo_reasoning.src.algorithms.sorting import insertion_sort, bubble_sort, heapsort, quicksort
from algo_reasoning.src.algorithms.greedy import activity_selector, task_scheduling
from algo_reasoning.src.algorithms.dynamic_programming import matrix_chain_order, lcs_length, optimal_bst
from algo_reasoning.src.algorithms.searching import minimum, binary_search, quickselect
from algo_reasoning.src.algorithms.divide_and_conquer import find_maximum_subarray_kadane
from algo_reasoning.src.algorithms.strings import naive_string_matcher, kmp_matcher
from algo_reasoning.src.algorithms.geometry import segments_intersect, graham_scan, jarvis_march
from algo_reasoning.src.algorithms.graphs import dfs, bfs, topological_sort, articulation_points, bridges, strongly_connected_components, mst_kruskal, mst_prim, bellman_ford, dijkstra, dag_shortest_paths, floyd_warshall

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
    "kmp_matcher": kmp_matcher, 
    "segments_intersect": segments_intersect,
    "graham_scan": graham_scan,
    "jarvis_march": jarvis_march,
    "dfs": dfs,
    "bfs": bfs,
    "topological_sort": topological_sort,
    "articulation_points": articulation_points,
    "bridges": bridges, 
    "strongly_connected_components": strongly_connected_components,
    "mst_kruskal": mst_kruskal,
    "mst_prim": mst_prim,
    "bellman_ford": bellman_ford,
    "dijkstra": dijkstra,
    "dag_shortest_paths": dag_shortest_paths,
    "floyd_warshall": floyd_warshall
}

class CLRS30Test(unittest.TestCase):

    def verify_data(self, out, sample, algo, ignore_keys=[]):
        for k in ignore_keys:
            if k in out.hints:
                del out.hints[k]
            if k in out.outputs:
                del out.outputs[k]

        self.assertCountEqual(out.inputs.keys(), sample.inputs.keys(), msg=f"Algo: {algo}, Inputs do not match")
        self.assertCountEqual(out.hints.keys(), sample.hints.keys(), msg=f"Algo: {algo}, Hints do not match")
        self.assertCountEqual(out.outputs.keys(), sample.outputs.keys(), msg=f"Algo: {algo}, Outputs do not match")

        self.assertEqual(out.algorithm, sample.algorithm, msg=f"Algo: {algo}, Algorithms do not match")
        
        def _compare_data(out_data, sample_data, hints=False):
            for k, value in out_data.items():

                if hints:
                    max_len = value.size(0)
                    
                    sample_value = sample_data[k][:max_len]
                else:
                    sample_value = sample_data[k]
                
                _, _, _type = SPECS[algo][k]

                if _type == Type.SCALAR:
                    assertion_tensor = torch.isclose(value, sample_value, rtol=1e-5, atol=1e-6)
                else:
                    assertion_tensor = (value == sample_value)

                self.assertTrue(
                        torch.all(assertion_tensor).item(),
                        msg=f"Key: {k}, Algo: {algo}, Tensors do not agree at indexes: {(~assertion_tensor).nonzero()}"
                    )
            
        _compare_data(out.inputs, sample.inputs)
        _compare_data(out.hints, sample.hints, hints=True)
        _compare_data(out.outputs, sample.outputs)
        

    def test_sorting(self):
        algorithms = ["insertion_sort",  "bubble_sort", "heapsort", "quicksort"]
        ds = OriginalCLRSDataset(algorithms, "train", "tmp/CLRS30")

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

            self.verify_data(out, sample, algo)

    def test_greedy(self):
        algorithms = ["activity_selector", "task_scheduling"]
        ds = OriginalCLRSDataset(algorithms, "train", "tmp/CLRS30")

        for i in range(len(ds)):
            sample = ds[i]
            sample.squeeze(inplace=True)

            inp = sample.inputs.clone().to_dict()
            
            nb_nodes = inp["pos"].size(0)
            algo = sample.algorithm

            del inp["pos"]

            out = algo_fn[algo](**inp, nb_nodes=nb_nodes)

            self.verify_data(out, sample, algo)

    def test_dynamic_programming(self):
        algorithms = ["matrix_chain_order", "lcs_length", "optimal_bst"]
        ds = OriginalCLRSDataset(algorithms, "train", "tmp/CLRS30")

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

            self.verify_data(out, sample, algo)

    def test_searching(self):
        algorithms = ["minimum", "binary_search", "quickselect"]
        ds = OriginalCLRSDataset(algorithms, "train", "tmp/CLRS30")

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

            self.verify_data(out, sample, algo)

    def test_divide_and_conquer(self):
        algorithms = ["find_maximum_subarray_kadane"]
        ds = OriginalCLRSDataset(algorithms, "train", "tmp/CLRS30")

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

            self.verify_data(out, sample, algo)

    def test_strings(self):
        algorithms = ["naive_string_matcher", "kmp_matcher"]

        ds = OriginalCLRSDataset(algorithms, "train", "tmp/CLRS30")

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

            self.verify_data(out, sample, algo)

    def test_geometry(self):
        algorithms = ["segments_intersect", "graham_scan", "jarvis_march"]

        ds = OriginalCLRSDataset(algorithms, "train", "tmp/CLRS30")

        for i in range(len(ds)):
            sample = ds[i]
            sample.squeeze(inplace=True)

            inp = sample.inputs.clone().to_dict()
            
            nb_nodes = inp["pos"].size(0)
            algo = sample.algorithm

            inp["xs"] = inp["x"].clone()
            inp["ys"] = inp["y"].clone()

            del inp["pos"]
            del inp["x"]
            del inp["y"]

            out = algo_fn[algo](**inp, nb_nodes=nb_nodes)

            self.verify_data(out, sample, algo)

    def test_graphs(self):
        algorithms = ["dfs", "bfs", "topological_sort", "articulation_points", "bridges", "strongly_connected_components", "mst_kruskal", "mst_prim", "bellman_ford", "dijkstra", "dag_shortest_paths", "floyd_warshall"]
        ds = OriginalCLRSDataset(algorithms, "train", "tmp/CLRS30")

        for i in range(len(ds)):
            sample = ds[i]
            sample.squeeze(inplace=True)

            inp = sample.inputs.clone().to_dict()
            
            nb_nodes = inp["pos"].size(0)
            algo = sample.algorithm

            if algo in ["bfs", "mst_prim", "bellman_ford", "dijkstra", "dag_shortest_paths"]:
                inp["s"] = torch.argmax(inp["s"]).item()

            del inp["pos"]
            del inp["adj"]

            out = algo_fn[algo](**inp, nb_nodes=nb_nodes)

            ignore_keys = ['s'] if algo == "dag_shortest_paths" else [] 

            self.verify_data(out, sample, algo, ignore_keys=ignore_keys)

if __name__ == '__main__':
    unittest.main()




