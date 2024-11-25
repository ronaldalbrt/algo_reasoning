import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from algo_reasoning.src.specs import SPECS, Type, OutputClass

REGULARIZATION_TYPES = ["constant_eigen", "tikhonov"]
    
class AlgorithmicReasoningLoss(nn.Module):
    def __init__(self, hint_loss_weight=0.1, reg_weight=0.0, reg_type="constant_eigen"):
        super().__init__()
        self.hint_loss = (hint_loss_weight > 0.0)
        self.hint_loss_weight = hint_loss_weight
        self.reg_weight = reg_weight
        self.reg_type = reg_type
        self.reg_term = self.reg_weight > 0.0 and reg_type in REGULARIZATION_TYPES
        
        if self.reg_term:
            if reg_type == "constant_eigen":
                self.regularizer = lambda hidden: torch.mean(torch.abs(torch.sum(hidden, dim=2)/(torch.norm(hidden, dim=2)*(math.sqrt(hidden.size(2))))))
            elif reg_type == "tikhonov":
                # Only considering complete graphs for nos
                def tikhonov_fourier_operator(embeddings, nb_nodes, device):
                    laplacian = torch.diag(torch.ones(nb_nodes, device=device)*nb_nodes) - torch.ones((nb_nodes, nb_nodes), device=device)
                    result = torch.linalg.eigh(laplacian)

                    eigenvectors = result.eigenvectors
                    fourier_embeddings = eigenvectors.T@embeddings

                    fourier_dist = torch.mean(torch.abs(fourier_embeddings), dim=-1)
                    fourier_dist_norm = torch.norm(fourier_dist, dim=2)

                    return -torch.mean(torch.sum(fourier_dist, dim=2)/(fourier_dist_norm*(math.sqrt(nb_nodes))))

                self.regularizer = tikhonov_fourier_operator

    def _calculate_loss(self, mask, truth, pred, type_, nb_nodes):
        if type_ == Type.SCALAR:
            return torch.mean(F.mse_loss(pred, truth, reduction='none') * mask)
        
        elif type_ == Type.MASK:
            masked_truth = (truth != OutputClass.MASKED)
            return torch.mean(F.binary_cross_entropy_with_logits(pred, truth, reduction='none') * mask * masked_truth)
        
        elif type_ in [Type.MASK_ONE, Type.CATEGORICAL]:
            masked_truth = truth * (truth != OutputClass.MASKED)

            logsoftmax_pred = F.log_softmax(pred, dim=-1)
            losses = logsoftmax_pred*mask*masked_truth

            return (-torch.sum(losses) / torch.sum(truth*mask == OutputClass.POSITIVE))
        
        elif type_ == Type.POINTER:
            
            return torch.mean(
                F.cross_entropy(
                    pred.transpose(-1, 1), 
                    F.one_hot(truth.long(), nb_nodes).float().transpose(-1, 1), 
                    reduction='none').transpose(1, -1) * mask.squeeze(-1))
        
        else:
            raise NotImplementedError
    
    def forward(self, pred, batch, hidden=None):
        algorithm = batch.algorithm
        specs = SPECS[algorithm]
        nb_nodes = batch.inputs.pos.shape[1]
        max_length = batch.max_length.long().item()
        device = batch.length.device

        if self.reg_term:
            assert hidden is not None, "Hidden Embeddings must be provided when reg_weight > 0.0"
            reg_loss = self.regularizer(hidden) if self.reg_type == "constant_eigen" else self.regularizer(hidden, nb_nodes, device)
        else:
            reg_loss = 0        

        output_loss = torch.zeros(1, device=device)
        for key, value in pred.outputs:
            if torch.isnan(value).any(): 
                raise Exception(f"NaN in {key} output")
            
            _, _, type_ = specs[key]
            mask = torch.ones_like(batch.outputs[key], device=device)

            output_loss += self._calculate_loss(mask, batch.outputs[key], value,  type_, nb_nodes)

        if self.hint_loss:
            hint_loss = torch.zeros(1, device=device)
            for key, value in pred.hints:
                _, _, type_ = specs[key]

                mask = torch.arange(max_length, device=device).unsqueeze(0) <= (batch.length - 1).unsqueeze(1)
                obj_dim = value.dim()
                #mask = mask[:, :, *[None for _ in range(obj_dim - 2)]]
                for _ in range(obj_dim - 2):
                    mask = mask.unsqueeze(-1)

                ground_truth = batch.hints[key][:, :max_length]

                hint_loss += self._calculate_loss(mask, ground_truth, value, type_, nb_nodes)

            output_loss += self.hint_loss_weight*hint_loss
            output_loss += self.reg_weight*reg_loss
        
        return output_loss