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
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import autograd

from algo_reasoning.src.specs import SPECS, Type, OutputClass
    
class AlgorithmicReasoningLoss(nn.Module):
    def __init__(self, hint_loss_weight=0.1, reg_weight=0.0):
        super().__init__()
        self.hint_loss = (hint_loss_weight > 0.0)
        self.hint_loss_weight = hint_loss_weight
        self.reg_weight = reg_weight
        self.reg_term = self.reg_weight > 0.0

        self.dummy_w =  nn.Parameter(torch.tensor(1.))
        
        if self.reg_term:
            self.regularizer = lambda emb: torch.mean(torch.abs(torch.sum(emb, dim=2)/(torch.norm(emb, dim=2)*(math.sqrt(emb.size(2))))))

    def _calculate_loss(self, mask, truth, pred, type_, nb_nodes):
        dim_to_reduce = list(range(1, pred.dim()))

        if type_ == Type.SCALAR:
            return torch.mean(F.mse_loss(pred, truth, reduction='none') * mask, dim=dim_to_reduce)
        
        elif type_ == Type.MASK:
            masked_truth = (truth != OutputClass.MASKED)
            return torch.mean(F.binary_cross_entropy_with_logits(pred, truth, reduction='none') * mask * masked_truth, dim=dim_to_reduce)
        
        elif type_ in [Type.MASK_ONE, Type.CATEGORICAL]:
            masked_truth = truth * (truth != OutputClass.MASKED)

            logsoftmax_pred = F.log_softmax(pred, dim=-1)
            losses = logsoftmax_pred*mask*masked_truth

            return (-torch.sum(losses, dim=dim_to_reduce) / torch.sum(truth*mask == OutputClass.POSITIVE, dim=dim_to_reduce))
        
        elif type_ == Type.POINTER:
            dim_to_reduce = dim_to_reduce[:-1]

            _cross_entropy = F.cross_entropy(
                    pred.transpose(-1, 1), 
                    F.one_hot(truth.long(), nb_nodes).float().transpose(-1, 1), 
                    reduction='none').transpose(1, -1) * mask.squeeze(-1)

            return torch.mean(_cross_entropy, dim=dim_to_reduce)
        
        else:
            raise NotImplementedError
    
    def forward(self, pred, batch, hidden=None, cur_epoch=None):
        algorithm = batch.algorithm
        specs = SPECS[algorithm]
        nb_nodes = batch.inputs.pos.shape[1]
        max_length = batch.max_length.long().item()
        device = batch.length.device

        output_loss = None
        for key, value in pred.outputs:
            if torch.isnan(value).any(): 
                raise Exception(f"NaN in {key} output")
            
            _, _, type_ = specs[key]
            mask = torch.ones_like(batch.outputs[key], device=device)

            if output_loss is None:
                output_loss = self._calculate_loss(mask, batch.outputs[key], value, type_, nb_nodes)
            else:
                output_loss += self._calculate_loss(mask, batch.outputs[key], value,  type_, nb_nodes)


        if self.hint_loss:
            hint_loss = None
            for key, value in pred.hints:
                _, _, type_ = specs[key]

                mask = torch.arange(max_length, device=device).unsqueeze(0) <= (batch.length - 1).unsqueeze(1)
                obj_dim = value.dim()

                for _ in range(obj_dim - 2):
                    mask = mask.unsqueeze(-1)

                ground_truth = batch.hints[key][:, :max_length]

                if hint_loss is None:
                    hint_loss = self._calculate_loss(mask, ground_truth, value, type_, nb_nodes)
                else:
                    hint_loss += self._calculate_loss(mask, ground_truth, value, type_, nb_nodes)

            output_loss += self.hint_loss_weight*hint_loss if hint_loss is not None else 0

        reg_weight = self.reg_weight
        if self.reg_term and self.training:
            reg_loss = self.regularizer(hidden)
            
        else:
            reg_loss = 0.0
        
        loss = output_loss.mean() + reg_weight*reg_loss
            
        return loss