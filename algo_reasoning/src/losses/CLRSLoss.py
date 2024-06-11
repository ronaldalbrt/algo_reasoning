import torch
import torch.nn as nn
import torch.nn.functional as F

from algo_reasoning.src.specs import Location, SPECS
from loguru import logger
    
class CLRSLoss(nn.Module):
    def __init__(self, hidden_loss_type, nb_nodes=16, hint_loss_weight=0.1):
        super().__init__()
        self.hint_loss = (hint_loss_weight > 0.0)
        self.hint_loss_weight = hint_loss_weight
        self.nb_nodes = nb_nodes

        # if hidden_loss_type == "l2":
        #     self.hidden_loss = lambda x: torch.mean(torch.linalg.norm(x, dim=1))
        # else:
        #     raise NotImplementedError(f"Unknown hidden loss type {hidden_loss_type}")
    
    def _calculate_loss(self, mask, truth, pred, type_):
        if type_ == "scalar":
            
            return torch.mean(F.mse_loss(pred, truth, reduction='none') * mask)
        
        elif type_ == "mask":
            
            return torch.mean(F.binary_cross_entropy_with_logits(pred, truth, reduction='none') * mask)
        
        elif type_ == "mask_one":
            logsoftmax_pred = F.log_softmax(pred, dim=-1)
            losses = truth*logsoftmax_pred*mask

            return losses.mean()
        
        elif type_ == "categorical":

            return torch.mean(
                F.cross_entropy(
                    pred.transpose(-1, 1), 
                    truth.transpose(-1, 1), 
                    reduction='none').transpose(-1, 1) * mask.squeeze(-1))
        
        elif type_ == "pointer":
            
            return torch.mean(
                F.cross_entropy(
                    pred.transpose(-1, 1), 
                    F.one_hot(truth.long(), self.nb_nodes).float().transpose(-1, 1), 
                    reduction='none').transpose(1, -1) * mask.squeeze(-1))
        
        else:
            raise NotImplementedError
    
    def forward(self, pred, batch):
        algorithm = batch.algorithm
        specs = SPECS[algorithm]
        max_length = batch.max_length.item()
        device = batch.length.device

        output_loss = torch.zeros(1, device=device)
        for key, value in pred.outputs:
            if torch.isnan(value).any(): 
                logger.warning(f"NaN in {key} output")
                raise Exception(f"NaN in {key} output")
            
            _, _, type_ = specs[key]
            mask = torch.ones_like(batch.outputs[key], device=device)
            output_loss += self._calculate_loss(mask, batch.outputs[key], value,  type_)

        hint_loss = torch.zeros(1, device=device)
        for key, value in pred.hints:
            
            if torch.isnan(value).any():
                logger.warning(f"NaN in {key} hint")
            _, _, type_ = specs[key]

            mask = torch.arange(max_length, device=device).unsqueeze(0) <= (batch.length - 1).unsqueeze(1)
            obj_dim = value.dim()
            mask = mask[:, :, *[None for _ in range(obj_dim - 2)]]

            ground_truth = batch.hints[key][:, :max_length]

            hint_loss += self._calculate_loss(mask, ground_truth, value, type_)
        
        return output_loss + (self.hint_loss_weight*hint_loss)