from typing import Dict, List, Tuple, Union

from algo_reasoning.src import specs
import torch
import numpy as np


_Location = specs.Location
_Stage = specs.Stage
_Type = specs.Type
_OutputClass = specs.OutputClass

_Array = np.ndarray | torch.Array
_Data = Union[_Array, List[_Array]]
_DataOrType = Union[_Data, str]

ProbesDict = Dict[str, Dict[str, Dict[str, Dict[str, _DataOrType]]]]

def initialize(spec: specs.Spec) -> ProbesDict:
  probes = dict()
  for stage in [_Stage.INPUT, _Stage.OUTPUT, _Stage.HINT]:
    probes[stage] = {}
    for loc in [_Location.NODE, _Location.EDGE, _Location.GRAPH]:
      probes[stage][loc] = {}

  for name in spec:
    stage, loc, t = spec[name]
    probes[stage][loc][name] = {}
    probes[stage][loc][name]['data'] = []
    probes[stage][loc][name]['type_'] = t

  return probes  # pytype: disable=bad-return-type
