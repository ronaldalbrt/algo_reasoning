# Copyright (C) 2024 Ronald Albert ronaldalbert@cos.ufrj.br

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#         http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import torch.nn.functional as F
from torchmetrics.functional import recall, precision, accuracy, auroc, f1_score
from algo_reasoning.src.specs import Type, OutputClass, SPECS, CATEGORIES_DIMENSIONS
from algo_reasoning.src.data import AlgorithmicData

from typing import Dict

def _preprocess_y(y, algorithm, key, type_, nb_nodes):
    num_classes = 2
    preprocesed_y = torch.clone(y)

    if type_ == Type.CATEGORICAL:
        num_classes = CATEGORIES_DIMENSIONS[algorithm][key]
        preprocesed_y = torch.argmax(preprocesed_y, dim=-1)
        
    elif type_ == Type.MASK_ONE:
        num_classes = nb_nodes
        preprocesed_y = torch.argmax(preprocesed_y, dim=-1)

    elif type_ == Type.POINTER:
        num_classes = nb_nodes
        
    return preprocesed_y.long(), num_classes

def _scalar_score(pred, y):
    return torch.mean(F.mse_loss(pred, y, reduction='none')).item()

def _multiclass_metrics(pred, y, num_classes, task, average="micro", ignore_index=None):
    """
    -------------------------------------------------------------------------------------------------
    Calculates the multiclass metrics for the model
    -------------------------------------------------------------------------------------------------
    Args:
        pred (torch.Tensor): the probability output of the model.
        y (torch.Tensor): the ground truth labels.
    -------------------------------------------------------------------------------------------------
    Returns: 
        The metrics dictionary (keys: roc_auc, acc, f1, precision, recall)
    -------------------------------------------------------------------------------------------------
    """
    def roc_auc(pred, y):
        return auroc(pred, y, task=task, num_classes=num_classes, ignore_index=ignore_index)
    
    def acc(pred, y):
        return accuracy(pred, y, task=task, num_classes=num_classes, average=average, ignore_index=ignore_index)
    
    def f1(pred, y):
        return f1_score(pred, y, task=task, num_classes=num_classes, average=average, ignore_index=ignore_index)
    
    def prec(pred, y):
        return precision(pred, y, task=task, num_classes=num_classes, average=average, ignore_index=ignore_index)
    
    def rec(pred, y):
        return recall(pred, y, task=task, num_classes=num_classes, average=average, ignore_index=ignore_index)


    return {
        'roc_auc': roc_auc(pred, y).item(), 
        'acc': acc(pred, y).item(), 
        'f1': f1(pred, y).item(), 
        'precision': prec(pred, y).item(),
        'recall': rec(pred, y).item()
        }

def _eval_on_values(value:torch.Tensor, 
                    y:torch.Tensor, 
                    type_:Type, 
                    output_metrics:Dict[str, float],
                    algorithm:str, 
                    key:str, 
                    average:str, 
                    nb_nodes:int):
    """
    -------------------------------------------------------------------------------------------------
    Evaluates the model's output on a given value.
    -------------------------------------------------------------------------------------------------
    Args:
        value (torch.Tensor): the model's output.
        y (torch.Tensor): the ground truth labels.
        type_ (Type): the type of the output.
        output_metrics (Dict[str, float]): the metrics dictionary.
        algorithm (str): the algorithm being evaluated.
        key (str): the key of the output.
        average (str): the averaging strategy to use for the metrics.
        nb_nodes (int): the number of nodes in the graph.
    -------------------------------------------------------------------------------------------------
    Returns:
        The updated metrics dictionary.
    -------------------------------------------------------------------------------------------------
    """
    if type_ == Type.SCALAR:
            score = _scalar_score(value, y)
            if "scalar_score" in output_metrics.keys():
                output_metrics["scalar_score"].append(score)
            else:
                output_metrics["scalar_score"] = [score]

    else:
        ignore_index = None
        if torch.any(y == OutputClass.MASKED):
                ignore_index = OutputClass.MASKED

        treated_y, num_classes = _preprocess_y(y, algorithm, key, type_, nb_nodes=nb_nodes)

        task = "binary" if type_ == Type.MASK else "multiclass"
        
        if value.dim() > 1:
            value = value.transpose(1, -1)
        
        score = _multiclass_metrics(value, treated_y, num_classes, task, average=average, ignore_index=ignore_index)

        for score_key in score.keys():
            if score_key in output_metrics.keys():
                output_metrics[score_key].append(score[score_key])
            else:
                output_metrics[score_key] = [score[score_key]]

    return output_metrics

def eval_function(pred:AlgorithmicData, 
                batch:AlgorithmicData, 
                average:str = "micro", 
                eval_hints:bool = True):
    """
    -------------------------------------------------------------------------------------------------
    Evaluates the model's predictions on a batch.
    -------------------------------------------------------------------------------------------------
    Args:
        pred (AlgorithmicData): the model's predictions.
        batch (AlgorithmicData): the batch to evaluate.
        average (str): the averaging strategy to use for the metrics.
        eval_hints (bool): whether to evaluate the hints.
    -------------------------------------------------------------------------------------------------
    Returns: 
        The metrics dictionary (keys: roc_auc, acc, f1, precision, recall)
    -------------------------------------------------------------------------------------------------
    """
    algorithm = batch.algorithm
    specs = SPECS[algorithm]
    nb_nodes = batch.inputs.pos.shape[1]

    output_metrics = {}
    for key, value in pred.outputs:
        _, _, type_ = specs[key]

        output_metrics = _eval_on_values(value, batch.outputs[key], type_, output_metrics, algorithm, key, average, nb_nodes)

    if eval_hints:
        for key, value in pred.hints:
            _, _, type_ = specs[key]

            y = batch.hints[key].flatten(end_dim=1)
            value = value.flatten(end_dim=1)

            output_metrics = _eval_on_values(value, y, type_, output_metrics, algorithm, key, average, nb_nodes)

    for score_key in output_metrics:
        output_metrics[score_key] = sum(output_metrics[score_key])/len(output_metrics[score_key])

    return output_metrics