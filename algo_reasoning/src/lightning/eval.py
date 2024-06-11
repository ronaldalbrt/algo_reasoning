import torch
from torcheval.metrics.functional import binary_accuracy, binary_f1_score, binary_precision,binary_recall, binary_auroc, binary_auprc
from torchmetrics.functional import jaccard_index

def metrics(prob, y):
    """
    # -------------------------------------------------------------------------------------------------
    # Calculates the metrics for the model
    # -------------------------------------------------------------------------------------------------
    # Args:
    # prob (torch.Tensor): the probability output of the model.
    # y (torch.Tensor): the ground truth labels.
    # -------------------------------------------------------------------------------------------------
    # Returns: 
    #   The metrics dictionary (keys: roc_auc, acc, f1, precision, recall)
    # -------------------------------------------------------------------------------------------------
    """
    def roc_auc(prob, y):
        return binary_auroc(prob, y)
    
    def accuracy(prob, y):
        return binary_accuracy(prob, y)
    
    def f1(prob, y):
        return binary_f1_score(prob, y)
    
    def precision(prob, y):
        return binary_precision(prob, y)
    
    def recall(prob, y):
        return binary_recall(prob.round().to(torch.bool), y.to(torch.bool))
    
    def iou(prob, y):
        return jaccard_index(prob.round().to(torch.bool), y.to(torch.bool), task="multiclass", num_classes=2)

    def auprc(prob, y):
        return binary_auprc(prob, y)

    return {
        'roc_auc': roc_auc(prob, y).item(), 
        'acc': accuracy(prob, y).item(), 
        'f1': f1(prob, y).item(), 
        'precision': precision(prob, y).item(),
        'recall': recall(prob, y).item(),
        'iou': iou(prob, y).item(),
        'auprc': auprc(prob, y).item()
        }
    