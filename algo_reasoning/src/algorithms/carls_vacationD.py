import torch
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


carls_vacation_specs = {
    "pos": (Stage.INPUT, Location.NODE, Type.SCALAR),
    'x1': (Stage.INPUT, Location.NODE, Type.SCALAR),
    'y1': (Stage.INPUT, Location.NODE, Type.SCALAR),
    'x2': (Stage.INPUT, Location.NODE, Type.SCALAR),
    'y2': (Stage.INPUT, Location.NODE, Type.SCALAR),
    'height1': (Stage.INPUT, Location.GRAPH, Type.SCALAR),
    'height2': (Stage.INPUT, Location.GRAPH, Type.SCALAR),
    'distance': (Stage.OUTPUT, Location.GRAPH, Type.SCALAR),
    }

def carls_vacation(x1, x2, y1, y2, height1, height2, nb_nodes):
  inputs = CLRSData()
  inputs['pos'] = ((torch.arange(nb_nodes) * 1.0)/nb_nodes).unsqueeze(0)

  inputs['x1'] = x1.float().unsqueeze(0)
  inputs['y1'] = y1.float().unsqueeze(0)
  inputs['x2'] = x2.float().unsqueeze(0)
  inputs['y2'] = y2.float().unsqueeze(0)

  inputs['height1'] = torch.tensor([height1]).float()
  inputs['height2'] = torch.tensor([height2]).float()

  

  
  
