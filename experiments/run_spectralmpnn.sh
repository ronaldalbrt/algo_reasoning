run_algorithm () {
    python run.py --algorithms $1 --model_name $1 --version_name spectralmpnn0 --processor_model spectralmpnn --batch_size $2
    python run.py --algorithms $1 --model_name $1 --version_name spectralmpnn1 --processor_model spectralmpnn --batch_size $2
    python run.py --algorithms $1 --model_name $1 --version_name spectralmpnn2 --processor_model spectralmpnn --batch_size $2
    python run.py --algorithms $1 --model_name $1 --version_name spectralmpnn3 --processor_model spectralmpnn --batch_size $2
    python run.py --algorithms $1 --model_name $1 --version_name spectralmpnn4 --processor_model spectralmpnn --batch_size $2
}

run_algorithm articulation_points 32
run_algorithm activity_selector 32
run_algorithm bellman_ford 32
run_algorithm bfs 32
run_algorithm binary_search 32
run_algorithm bridges 32
run_algorithm bubble_sort 32
run_algorithm dag_shortest_paths 32
run_algorithm dfs 32
run_algorithm dijkstra 32
run_algorithm find_maximum_subarray_kadane 32
run_algorithm floyd_warshall 16
run_algorithm graham_scan 32
run_algorithm heapsort 32
run_algorithm insertion_sort 32
run_algorithm jarvis_march 16
run_algorithm kmp_matcher 32
run_algorithm lcs_length 32
run_algorithm matrix_chain_order 16
run_algorithm minimum 32
run_algorithm mst_kruskal 32
run_algorithm mst_prim 32
run_algorithm naive_string_matcher 32
run_algorithm optimal_bst 16
run_algorithm quickselect 32
run_algorithm quicksort 32
run_algorithm segments_intersect 32
run_algorithm strongly_connected_components 32
run_algorithm task_scheduling 32
run_algorithm topological_sort 32