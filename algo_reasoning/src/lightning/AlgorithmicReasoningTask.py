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

import torch.nn.functional as F
import lightning as L
from algo_reasoning.src.eval import eval_function
from lightning.pytorch.utilities import grad_norm


class AlgorithmicReasoningTask(L.LightningModule):
    def __init__(self, model, loss_fn, optim_method, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=['model','loss_fn'])
        
        self.model = model
        self.loss_fn = loss_fn
        self.optim_method = optim_method
        self.lr = kwargs.get('lr', 1e-3)
        self.weight_decay = kwargs.get('weight_decay', 1e-2)

    def _batch_loss(self, batch, calculate_metrics=False, prefix="val"):
        input_batch = batch.clone()
        output = self.model(input_batch)

        preds = output["output"]
        hidden = output["hidden_embeddings"]

        loss = self.loss_fn(preds, batch, hidden)

        if calculate_metrics:
            metrics_r = eval_function(preds.detach().cpu(), batch.detach().cpu())
            metrics_output = eval_function(preds.detach().cpu(), batch.detach().cpu(), eval_hints=False)
            metrics_output = {k+'_output': v for k, v in metrics_output.items()}
            
            metrics_r["loss"] = loss

            metrics = {**metrics_r, **metrics_output}

            return {f'{prefix}_{k}': v for k, v in metrics.items()}
        
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._batch_loss(batch)

        self.log("train_loss", loss)

        return loss
    

    def validation_step(self, batch, batch_idx):
        metrics = self._batch_loss(batch, calculate_metrics=True, prefix="val")

        self.log_dict(metrics, sync_dist=True)

        return metrics["val_loss"]
    
    
    def test_step(self, batch, batch_idx):
        metrics = self._batch_loss(batch, calculate_metrics=True, prefix="test")

        self.log_dict(metrics, sync_dist=True)

        return metrics["test_loss"]

    def on_before_optimizer_step(self, optimizer):
        norms = grad_norm(self.model, norm_type=2)
        self.log_dict(norms)

    def configure_optimizers(self):
        optimizer = self.optim_method(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        print(optimizer.state_dict)
        return optimizer