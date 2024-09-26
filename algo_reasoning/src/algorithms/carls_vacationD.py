import os
import torch
import torch.linalg as LA
import math
from algo_reasoning.src.data import CLRSData
from algo_reasoning.src.specs import Stage, Location, Type

# GEOMETRY / SEGMENTS INTERSECT
def segments_intersect(xs, ys):
  """Segment intersection."""

  dirs = torch.zeros(xs.shape[0])
  on_seg = torch.zeros(xs.shape[0])

  def cross_product(x1, y1, x2, y2):
    return x1 * y2 - x2 * y1

  def direction(xs, ys, i, j, k):
    return cross_product(xs[k] - xs[i], ys[k] - ys[i], xs[j] - xs[i],
                         ys[j] - ys[i])

  def on_segment(xs, ys, i, j, k):
    if min(xs[i], xs[j]) <= xs[k] and xs[k] <= max(xs[i], xs[j]):
      if min(ys[i], ys[j]) <= ys[k] and ys[k] <= max(ys[i], ys[j]):
        return 1
    return 0

  dirs[0] = direction(xs, ys, 2, 3, 0)
  on_seg[0] = on_segment(xs, ys, 2, 3, 0)

  dirs[1] = direction(xs, ys, 2, 3, 1)
  on_seg[1] = on_segment(xs, ys, 2, 3, 1)

  dirs[2] = direction(xs, ys, 0, 1, 2)
  on_seg[2] = on_segment(xs, ys, 0, 1, 2)

  dirs[3] = direction(xs, ys, 0, 1, 3)
  on_seg[3] = on_segment(xs, ys, 0, 1, 3)

  ret = 0

  if ((dirs[0] > 0 and dirs[1] < 0) or
      (dirs[0] < 0 and dirs[1] > 0)) and ((dirs[2] > 0 and dirs[3] < 0) or
                                          (dirs[2] < 0 and dirs[3] > 0)):
    ret = 1
  elif dirs[0] == 0 and on_seg[0]:
    ret = 1
  elif dirs[1] == 0 and on_seg[1]:
    ret = 1
  elif dirs[2] == 0 and on_seg[2]:
    ret = 1
  elif dirs[3] == 0 and on_seg[3]:
    ret = 1

  return ret

def segments_distance(xs, ys):
  """ distance between two segments in the plane:
      one segment is (xs[0], ys[0]) to (xs[1], ys[1])
      the other is   (xs[2], ys[2]) to (xs[3], ys[3])
  """
  if segments_intersect(xs, ys): return 0
  # try each of the 4 vertices w/the other segment
  distances = []
  distances.append(point_segment_distance(xs[0], ys[0], xs[2], ys[2], xs[3], ys[3]))
  distances.append(point_segment_distance(xs[1], ys[1], xs[2], ys[2], xs[3], ys[3]))
  distances.append(point_segment_distance(xs[2], ys[2], xs[0], ys[0], xs[1], ys[1]))
  distances.append(point_segment_distance(xs[3], ys[3], xs[0], ys[0], xs[1], ys[1]))
  return min(distances)

def point_segment_distance(px, py, x1, y1, x2, y2):
  dx = x2 - x1
  dy = y2 - y1
  if dx == dy == 0:  # the segment's just a point
    return math.hypot(px - x1, py - y1)

  # Calculate the t that minimizes the distance.
  t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)

  # See if this represents one of the segment's
  # end points or a point in the middle.
  if t < 0:
    dx = px - x1
    dy = py - y1
  elif t > 1:
    dx = px - x2
    dy = py - y2
  else:
    near_x = x1 + t * dx
    near_y = y1 + t * dy
    dx = px - near_x
    dy = py - near_y

  return math.hypot(dx, dy)

def square_from_segment(p1, p2):
  dx1 = p2[0] - p1[0]
  dy1 = p2[1] - p1[1]

  mov1 = torch.tensor([-dy1, dx1])
  
  p3 = p2 + mov1
  
  dx2 = p3[0] - p2[0]
  dy2 = p3[1] - p2[1]

  mov2 = torch.tensor([-dy2, dx2 ])

  p4 = p3 + mov2

  return torch.stack([p1, p2, p4, p3])

def generate_non_intersecting_squares():
  p3 = ((torch.rand((2)) * (10**3 + 10**2)) - 10**2)
  p4 = p3 + torch.rand((2)) * 10**2

  distance = LA.vector_norm(p3 - p4).item()
  diameter = distance * math.sqrt(2)
  
  p1 = (p3 - diameter) - torch.rand((2)) * 10**2
  p2 = p1 - torch.rand((2)) * 10**2

  height1, height2 = ((torch.rand((2)) * (10**2 + 10**2)) - 10**2)

  x = torch.tensor([p1[0], p2[0], p3[0], p4[0]])
  y = torch.tensor([p1[1], p2[1], p3[1], p4[1]])

  return x, y, height1.item(), height2.item()


def cross_product(a, b):
  return (a[0]*b[1] - a[1]*b[0]).item()

carls_vacation_specs = {
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

def carls_vacation(x, y, height1, height2, nb_nodes):
  inputs = CLRSData()
  length = 1
  inputs['pos'] = ((torch.arange(nb_nodes) * 1.0)/nb_nodes).unsqueeze(0)

  inputs['x'] = x.float().unsqueeze(0)
  inputs['y'] = y.float().unsqueeze(0)

  inputs['height1'] = torch.tensor([height1]).float()
  inputs['height2'] = torch.tensor([height2]).float()

  p1, p2, p3, p4 = (torch.tensor([x[0], y[0]]), torch.tensor([x[1], y[1]]), torch.tensor([x[2], y[2]]), torch.tensor([x[3], y[3]]))

  faces1 = square_from_segment(p1, p2)
  faces2 = square_from_segment(p3, p4)

  top1 = torch.cat((torch.mean(faces1, dim=0), torch.tensor([height1])), dim=0)
  top2 = torch.cat((torch.mean(faces2, dim=0), torch.tensor([height2])), dim=0)

  hints = CLRSData()
  hints['faces1_x'] = faces1[:, 0].float().unsqueeze(0).unsqueeze(0)
  hints['faces1_y'] = faces1[:, 1].float().unsqueeze(0).unsqueeze(0)
  hints['faces2_x'] = faces2[:, 0].float().unsqueeze(0).unsqueeze(0)
  hints['faces2_y'] = faces2[:, 1].float().unsqueeze(0).unsqueeze(0)
  hints['tops_segment1_x'] = torch.zeros(1, 1, nb_nodes)
  hints['tops_segment1_y'] = torch.zeros(1, 1, nb_nodes)
  hints['tops_segment2_x'] = torch.zeros(1, 1, nb_nodes)
  hints['tops_segment2_y'] = torch.zeros(1, 1, nb_nodes)
  hints['selected_segment1'] = torch.zeros(1, 1, nb_nodes)
  hints['selected_segment2'] = torch.zeros(1, 1, nb_nodes)

  min_distance = float('inf')
  for i in range(nb_nodes):
    segment1 = faces1[[(i % nb_nodes), ((i + 1) % nb_nodes)]]
    mo1 = torch.mean(segment1, dim=0) 
    p1 = (segment1[0] - segment1[1])[[1, 0]]
    p1[1] = -p1[1]
    mid1 = mo1 + p1 / 2

    for j in range(nb_nodes):
      segment2 = faces2[[(j % nb_nodes), ((j + 1) % nb_nodes)]]
      mo2 = torch.mean(segment2, dim=0)
      p2 = (segment2[0] - segment2[1])[[1, 0]]
      p2[1] = -p2[1]
      mid2 = mo2 + p2 / 2

      for diag1 in range(2):
        len1 = LA.vector_norm(torch.cat((segment1[0], torch.tensor([0]))) - top1).item() if diag1 else 0.0

        top1_mo1 = top1 - torch.cat((mo1, torch.tensor([0])))
        top_mo_distance1 = LA.vector_norm(top1_mo1)
        unit_vector1 = top1_mo1[:2] / LA.vector_norm(top1_mo1[:2])

        rotated_top1 = mo1[:2] + unit_vector1 * top_mo_distance1
          
        at = segment1[0] if diag1 else rotated_top1
        
        for diag2 in range(2):
          len2 = LA.vector_norm(torch.cat((segment2[0], torch.tensor([0]))) - top2).item()  if diag2 else 0.0

          top2_mo2 = top2 - torch.cat((mo2, torch.tensor([0])))
          top_mo_distance2 = LA.vector_norm(top2_mo2)
          unit_vector2 = top2_mo2[:2] / LA.vector_norm(top2_mo2[:2])

          rotated_top2 = mo2[:2] + unit_vector2 * top_mo_distance2

          bt = segment2[0] if diag2 else rotated_top2

          tops_segment1_x = torch.cat((torch.tensor([at[0], bt[0]]), segment1[:, 0]))
          tops_segment1_y = torch.cat((torch.tensor([at[1], bt[1]]), segment1[:, 1]))

          tops_segment2_x = torch.cat((torch.tensor([at[0], bt[0]]), segment2[:, 0]))
          tops_segment2_y = torch.cat((torch.tensor([at[1], bt[1]]), segment2[:, 1]))
        
          current_distance = len1 + len2 + LA.vector_norm(at - bt).item()

          if (not diag1) and (cross_product(mid2-segment1[0], segment1[1]-segment1[0]) < 0 or not segments_intersect(tops_segment1_x, tops_segment1_y)):
            continue
          if (not diag2) and (cross_product(mid1-segment2[0], segment2[1]-segment2[0]) < 0 or not segments_intersect(tops_segment2_x, tops_segment2_y)):
            continue
          else:
            if min_distance > current_distance:
              min_distance = current_distance
              selected_segment1 = i
              selected_segment2 = j
              selected_tops_segment1_x = tops_segment1_x
              selected_tops_segment1_y = tops_segment1_y
              selected_tops_segment2_x = tops_segment2_x
              selected_tops_segment2_y = tops_segment2_y

  aranged_nb_nodes = torch.arange(nb_nodes)
  aranged_selected_segment1 = torch.isin(aranged_nb_nodes, torch.tensor([(selected_segment1 % nb_nodes), ((selected_segment1 + 1) % nb_nodes)]))
  aranged_selected_segment2 = torch.isin(aranged_nb_nodes, torch.tensor([(selected_segment2 % nb_nodes), ((selected_segment2 + 1) % nb_nodes)]))
  length += 1

  hints['faces1_x'] = torch.cat((hints['faces1_x'], faces1[:, 0].float().unsqueeze(0).unsqueeze(0)), 1)
  hints['faces1_y'] = torch.cat((hints['faces1_x'], faces1[:, 1].float().unsqueeze(0).unsqueeze(0)), 1)
  hints['faces2_x'] = torch.cat((hints['faces2_x'], faces2[:, 0].float().unsqueeze(0).unsqueeze(0)), 1)
  hints['faces2_y'] = torch.cat((hints['faces2_y'], faces2[:, 1].float().unsqueeze(0).unsqueeze(0)), 1)
  hints['tops_segment1_x'] = torch.cat((hints['tops_segment1_x'], selected_tops_segment1_x.unsqueeze(0).unsqueeze(0)), 1)
  hints['tops_segment1_y'] = torch.cat((hints['tops_segment1_y'], selected_tops_segment1_y.unsqueeze(0).unsqueeze(0)), 1)
  hints['tops_segment2_x'] = torch.cat((hints['tops_segment2_x'], selected_tops_segment2_x.unsqueeze(0).unsqueeze(0)), 1)
  hints['tops_segment2_y'] = torch.cat((hints['tops_segment2_y'], selected_tops_segment2_y.unsqueeze(0).unsqueeze(0)), 1)
  hints['selected_segment1'] = torch.cat((hints['selected_segment1'], aranged_selected_segment1.unsqueeze(0).unsqueeze(0)), 1)
  hints['selected_segment2'] = torch.cat((hints['selected_segment2'], aranged_selected_segment2.unsqueeze(0).unsqueeze(0)), 1)

  outputs = CLRSData()
  outputs['distance'] = torch.tensor([min_distance]).float()

  return CLRSData(inputs=inputs, hints=hints, length=torch.tensor(length).float(), outputs=outputs, algorithm="carls_vacation")

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
    for _ in range(32):
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
    for _ in range(32):
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
  

  
  
