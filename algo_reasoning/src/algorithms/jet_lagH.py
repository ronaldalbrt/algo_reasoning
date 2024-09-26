import os
import torch
import torch.linalg as LA
import math
from algo_reasoning.src.data import CLRSData
from algo_reasoning.src.specs import Stage, Location, Type

jet_lag_specs = {
    "pos": (Stage.INPUT, Location.NODE, Type.SCALAR),
    'x': (Stage.INPUT, Location.NODE, Type.SCALAR),
    'y': (Stage.INPUT, Location.NODE, Type.SCALAR),
    'height1': (Stage.INPUT, Location.GRAPH, Type.SCALAR),
    'height2': (Stage.INPUT, Location.GRAPH, Type.SCALAR),
    'distance': (Stage.OUTPUT, Location.GRAPH, Type.SCALAR),
    'faces1_x': (Stage.HINT, Location.NODE, Type.SCALAR),
    'faces1_y': (Stage.HINT, Location.NODE, Type.SCALAR),
    'faces2_x': (Stage.HINT, Location.NODE, Type.SCALAR),
    'faces2_y': (Stage.HINT, Location.NODE, Type.SCALAR),
    'tops_segment1_x': (Stage.HINT, Location.NODE, Type.SCALAR),
    'tops_segment1_y': (Stage.HINT, Location.NODE, Type.SCALAR),
    'tops_segment2_x': (Stage.HINT, Location.NODE, Type.SCALAR),
    'tops_segment2_y': (Stage.HINT, Location.NODE, Type.SCALAR),
    'faces2_y': (Stage.HINT, Location.NODE, Type.SCALAR),
    'selected_segment1': (Stage.HINT, Location.NODE, Type.MASK),
    'selected_segment2': (Stage.HINT, Location.NODE, Type.MASK),
    }

def jet_lag(x, y, height1, height2, nb_nodes):
  
  return nb_nodes

if __name__ == "__main__":
    os.mkdir("tmp/CLRS30/carls_vacation")
    os.mkdir("tmp/CLRS30/carls_vacation/train")

    # Sampling Training set
    x_train = []
    y_train = []
    height1_train = []
    height2_train = []
    for _ in range(1000):
        x, y, height1, height2 = generate_non_intersecting_squares()
        x_train.append(x)
        y_train.append(y)
        height1_train.append(height1)
        height2_train.append(height2)

    train_datapoints = []
    max_length = -1
    for x, y, height1, height2 in zip(x_train, y_train, height1_train, height2_train):
        nb_nodes = 4

        data_point = carls_vacation(x, y, height1, height2, nb_nodes)
        train_datapoints.append(data_point)
        curr_length = data_point.length.item()
        max_length = curr_length if curr_length > max_length else max_length


    os.mkdir("tmp/CLRS30/carls_vacation/val")
    val_datapoints = []
    # Sampling Validation set
    x_val = []
    y_val = []
    height1_val = []
    height2_val = []
    for _ in range(1000):
        x, y, height1, height2 = generate_non_intersecting_squares()
        x_val.append(x)
        y_val.append(y)
        height1_val.append(height1)
        height2_val.append(height2)

    for x, y, height1, height2 in zip(x_val, y_val, height1_val, height2_val):
        nb_nodes = 4

        data_point = carls_vacation(x, y, height1, height2, nb_nodes)
        val_datapoints.append(data_point)
        curr_length = data_point.length.item()
        max_length = curr_length if curr_length > max_length else max_length


    os.mkdir("tmp/CLRS30/carls_vacation/test")
    test_datapoints = []
    # Sampling Test set
    x_test = []
    y_test = []
    height1_test = []
    height2_test = []
    for _ in range(1000):
        x, y, height1, height2 = generate_non_intersecting_squares()
        x_test.append(x)
        y_test.append(y)
        height1_test.append(height1)
        height2_test.append(height2)

    for x, y, height1, height2 in zip(x_val, y_val, height1_val, height2_val):
        nb_nodes = 4

        data_point = carls_vacation(x, y, height1, height2, nb_nodes)
        test_datapoints.append(data_point)
        curr_length = data_point.length.item()
        max_length = curr_length if curr_length > max_length else max_length

    for i, data_point in enumerate(train_datapoints):
        data_point["max_length"] = torch.tensor(max_length)
        torch.save(data_point, f"tmp/CLRS30/carls_vacation/train/{i}")

    for i, data_point in enumerate(val_datapoints):
        data_point["max_length"] = torch.tensor(max_length)
        torch.save(data_point, f"tmp/CLRS30/carls_vacation/val/{i}")

    for i, data_point in enumerate(test_datapoints):
        data_point["max_length"] = torch.tensor(max_length)
        torch.save(data_point, f"tmp/CLRS30/carls_vacation/test/{i}")