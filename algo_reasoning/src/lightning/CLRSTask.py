import os
import torch
import torch.nn.functional as F
import lightning as L
from algo_reasoning.src.lightning.eval import metrics


class CLRSTask(L.LightningModule):
    def __init__(self, model, loss_fn, optim_method, lr):
        super().__init__()
        self.save_hyperparameters(ignore=['model','loss_fn'])
        
        self.model = model
        self.loss_fn = loss_fn
        self.optim_method = optim_method
        self.lr = lr

    def _batch_loss(self, batch, calculate_metrics=False, prefix="val"):
        preds = self.model(batch)

        loss = self.loss_fn(preds, batch)

        if calculate_metrics:
            pred_prob = F.sigmoid(preds)
            metrics_r = metrics(torch.flatten(pred_prob).detach().cpu(), torch.flatten(labels).detach().cpu())
            
            metrics_r["loss"] = loss

            return {f'{prefix}_{k}': v for k, v in metrics_r.items()}
        
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._batch_loss(batch)

        self.log("train_loss", loss)

        return loss
    

    def validation_step(self, batch, batch_idx):
        #metrics = self._batch_loss(batch, prefix="val")

        #self.log_dict(metrics, sync_dist=True)

        #return metrics["val_loss"]
        loss = self._batch_loss(batch)

        self.log("val_loss", loss)

        return loss
    
    
    def test_step(self, batch, batch_idx):
        #metrics = self._batch_loss(batch, prefix="test")

        #self.log_dict(metrics, sync_dist=True)

        #return metrics["test_loss"]

        loss = self._batch_loss(batch)

        self.log("val_loss", loss)

        return loss

    def configure_optimizers(self):
        optimizer = self.optim_method(self.parameters(), lr=self.lr)
        return optimizer