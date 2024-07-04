import itertools
import os
import torch
from algo_reasoning.src.data import CLRSData
from algo_reasoning.src.specs import Stage, Location, Type
import networkx as nx

vertex_cover_specs = {
    "pos": (Stage.INPUT, Location.NODE, Type.SCALAR),
    'A': (Stage.INPUT, Location.EDGE, Type.SCALAR),
    'adj': (Stage.INPUT, Location.EDGE, Type.MASK),
    'in_cover':(Stage.OUTPUT, Location.NODE, Type.MASK),
    'min_vertex_cover': (Stage.HINT, Location.GRAPH, Type.SCALAR),
}

def validity_check(graph, cover):
        is_valid = True
        for i in range(len(graph)):
            for j in range(i+1, len(graph[i])):
                if graph[i][j] == 1 and cover[i] != 1 and cover[j] != 1:
                    return False

        return is_valid

def vertex_cover_naive(graph):
    nb_nodes = len(graph)
    minimum_vertex_cover = nb_nodes

    inputs = CLRSData()
    inputs["adj"] = graph.unsqueeze(0).float()
    inputs["A"] = graph.unsqueeze(0).float()
    inputs["pos"] = ((torch.arange(nb_nodes) * 1.0)/nb_nodes).unsqueeze(0)
    length = 1

    min_cover_hint = torch.tensor([minimum_vertex_cover])
    
    a = iter(list(i) for i in itertools.product([0, 1], repeat=nb_nodes) if validity_check(graph, i))
    for i in a:
        counter = 0
        for value in i:
            if value == 1:
                counter += 1
        
        if counter < minimum_vertex_cover:
            length += 1
            minimum_vertex_cover = counter
            min_cover_hint = torch.cat((min_cover_hint, torch.tensor([minimum_vertex_cover])))
            cover = torch.tensor([i])

    hints = CLRSData()
    hints["min_vertex_cover"] = min_cover_hint.unsqueeze(0).float()
    
    outputs = CLRSData()
    outputs["in_cover"] = cover.float()

    return CLRSData(inputs=inputs, hints=hints, length=torch.tensor(length).float(), outputs=outputs, algorithm="naive_vertex_cover")

if __name__ == "__main__":
    os.mkdir("tmp/CLRS30/naive_vertex_cover")
    
    os.mkdir("tmp/CLRS30/naive_vertex_cover/train")

    # Sampling Training set
    p_train = torch.rand((1000,)).tolist()

    train_datapoints = []
    max_length = -1
    for p in p_train:
        graph = torch.tensor(nx.adjacency_matrix(nx.erdos_renyi_graph(8, p)).toarray())

        data_point = vertex_cover_naive(graph)
        train_datapoints.append(data_point)
        curr_length = data_point.length.long().item()
        max_length = curr_length if curr_length > max_length else max_length


    os.mkdir("tmp/CLRS30/naive_vertex_cover/val")
    val_datapoints = []
    # Sampling Validation set
    p_val = torch.rand((32,)).tolist()

    for p in p_val:
        graph = torch.tensor(nx.adjacency_matrix(nx.erdos_renyi_graph(8, p)).toarray())

        data_point = vertex_cover_naive(graph)
        val_datapoints.append(data_point)
        curr_length = data_point.length.long().item()
        max_length = curr_length if curr_length > max_length else max_length


    os.mkdir("tmp/CLRS30/naive_vertex_cover/test")
    test_datapoints = []
    # Sampling Test set
    p_test = torch.rand((32,)).tolist()

    for p in p_test:
        graph = torch.tensor(nx.adjacency_matrix(nx.erdos_renyi_graph(16, p)).toarray())

        data_point = vertex_cover_naive(graph)
        test_datapoints.append(data_point)
        curr_length = data_point.length.long().item()
        max_length = curr_length if curr_length > max_length else max_length

    for i, data_point in enumerate(train_datapoints):
        _, dp_length = data_point.hints.min_vertex_cover.shape

        last_value = data_point.hints.min_vertex_cover[0, -1]
        data_point["max_length"] = max_length
        data_point.hints.min_vertex_cover = torch.cat((data_point.hints.min_vertex_cover, torch.full((1, int(max_length) - dp_length), last_value)), dim=1)
        torch.save(data_point, f"tmp/CLRS30/naive_vertex_cover/train/{i}")

    for i, data_point in enumerate(val_datapoints):
        dp_length = data_point.hints.min_vertex_cover.shape[1]

        last_value = data_point.hints.min_vertex_cover[0, -1]
        data_point["max_length"] = max_length
        data_point.hints.min_vertex_cover = torch.cat((data_point.hints.min_vertex_cover, torch.full((1, int(max_length) - dp_length), last_value)), dim=1)
        torch.save(data_point, f"tmp/CLRS30/naive_vertex_cover/val/{i}")

    for i, data_point in enumerate(test_datapoints):
        _, dp_length = data_point.hints.min_vertex_cover.shape

        last_value = data_point.hints.min_vertex_cover[0, -1]
        data_point["max_length"] = max_length
        data_point.hints.min_vertex_cover = torch.cat((data_point.hints.min_vertex_cover, torch.full((1, int(max_length) - dp_length), last_value)), dim=1)
        torch.save(data_point, f"tmp/CLRS30/naive_vertex_cover/test/{i}")