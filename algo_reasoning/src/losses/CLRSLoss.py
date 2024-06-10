import torch
import torch.nn.functional as F

from algo_reasoning.src.data.specs import Location, SPECS
from loguru import logger

def calculate_loss(mask, truth, pred, type_):
    if type_ == "scalar":
        
        return torch.mean(F.mse_loss(pred, truth, reduction='none') * mask)
    
    elif type_ == "mask":
        
        return torch.mean(F.binary_cross_entropy_with_logits(pred, truth, reduction='none') * mask)
    
    elif type_ == "mask_one":
        logsoftmax_pred = F.log_softmax(pred, dim=-1)
        losses = truth*logsoftmax_pred*mask

        return losses.mean()
    
    elif type_ == "categorical":
        # TODO: Check if this is correct
        categories = pred.shape[-1]

        mask = mask.unsqueeze(-1).repeat_interleave(categories, dim=-1) # B x N -> B x N x C
        return torch.mean(F.cross_entropy(pred, truth, reduction='none') * mask)
    
    elif type_ == "pointer":
        return torch.mean(F.cross_entropy(pred, truth, reduction='none') * mask)
    else:
        raise NotImplementedError
    
class CLRSLoss(torch.nn.Module):
    def __init__(self, hidden_loss_type, hint_loss=True):
        super().__init__()
        self.hint_loss = hint_loss

        # if hidden_loss_type == "l2":
        #     self.hidden_loss = lambda x: torch.mean(torch.linalg.norm(x, dim=1))
        # else:
        #     raise NotImplementedError(f"Unknown hidden loss type {hidden_loss_type}")

    def forward(self, pred, batch):
        algorithm = batch.algorithm
        specs = SPECS[algorithm]
        max_length = batch.length.max()
        device = batch.length.device


        output_loss = torch.zeros(1, device=device)
        for key, value in pred.outputs:
            # check of nan
            if torch.isnan(value).any(): 
                logger.warning(f"NaN in {key} output")
                raise Exception(f"NaN in {key} output")
            
            _, _, type_, _ = specs[key]
            mask = torch.ones_like(batch.outputs[key], device=device)
            output_loss += calculate_loss(mask, batch.outputs[key], value,  type_)

        hint_loss = torch.zeros(1, device=device)
        for key, value in pred.hints:
            # check of nan
            if torch.isnan(value).any():
                logger.warning(f"NaN in {key} hint")
            _, loc, type_, _ = self.specs[key]

            # TODO: Check if this is correct
            mask = torch.ones_like(batch[key], device=device)
            mask[:, :] = torch.arange(max_length, device=device).unsqueeze(0) <= (batch.length - 1).unsqueeze(1)
            
            hint_loss += calculate_loss(mask, batch[key], value, type_)
        
        return output_loss, hint_loss