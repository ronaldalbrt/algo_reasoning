import torch
import torch.nn as nn
import torch.nn.functional as F

from algo_reasoning.src.specs import SPECS, Type
from loguru import logger
    
class CLRSLoss(nn.Module):
    def __init__(self, hint_loss_weight=0.1):
        super().__init__()
        self.hint_loss = (hint_loss_weight > 0.0)
        self.hint_loss_weight = hint_loss_weight
    
    def _calculate_loss(self, mask, truth, pred, type_, nb_nodes):
        if type_ == Type.SCALAR:
            return torch.mean(F.mse_loss(pred, truth, reduction='none') * mask)
        
        elif type_ == Type.MASK:
            return torch.mean(F.binary_cross_entropy_with_logits(pred, truth, reduction='none') * mask)
        
        elif type_ == Type.MASK_ONE:
            logsoftmax_pred = F.log_softmax(pred, dim=-1)
            losses = truth*logsoftmax_pred*mask

            losses *= -1

            return losses.mean()
        
        elif type_ == Type.CATEGORICAL:

            return torch.mean(
                F.cross_entropy(
                    pred.transpose(-1, 1), 
                    truth.transpose(-1, 1), 
                    reduction='none').transpose(-1, 1) * mask.squeeze(-1))
        
        elif type_ == Type.POINTER:
            
            return torch.mean(
                F.cross_entropy(
                    pred.transpose(-1, 1), 
                    F.one_hot(truth.long(), nb_nodes).float().transpose(-1, 1), 
                    reduction='none').transpose(1, -1) * mask.squeeze(-1))
        
        else:
            raise NotImplementedError
    
    def forward(self, pred, batch):
        algorithm = batch.algorithm
        specs = SPECS[algorithm]
        nb_nodes = batch.inputs.pos.shape[1]
        max_length = batch.max_length.item()
        device = batch.length.device

        output_loss = torch.zeros(1, device=device)
        for key, value in pred.outputs:
            if torch.isnan(value).any(): 
                logger.warning(f"NaN in {key} output")
                raise Exception(f"NaN in {key} output")
            
            _, _, type_ = specs[key]
            mask = torch.ones_like(batch.outputs[key], device=device)

            output_loss += self._calculate_loss(mask, batch.outputs[key], value,  type_, nb_nodes)

        if self.hint_loss:
            hint_loss = torch.zeros(1, device=device)
            for key, value in pred.hints:
                
                if torch.isnan(value).any():
                    logger.warning(f"NaN in {key} hint")
                _, _, type_ = specs[key]

                mask = torch.arange(max_length, device=device).unsqueeze(0) <= (batch.length - 1).unsqueeze(1)
                obj_dim = value.dim()
                mask = mask[:, :, *[None for _ in range(obj_dim - 2)]]

                ground_truth = batch.hints[key][:, :max_length]

                hint_loss += self._calculate_loss(mask, ground_truth, value, type_, nb_nodes)

            output_loss += self.hint_loss_weight*hint_loss
        
        return output_loss