import torch.nn.functional as F
import lightning as L
from algo_reasoning.src.eval import eval_function
from lightning.pytorch.utilities import grad_norm


class AlgorithmicReasoningTask(L.LightningModule):
    def __init__(self, model, loss_fn, optim_method, lr):
        super().__init__()
        self.save_hyperparameters(ignore=['model','loss_fn'])
        
        self.model = model
        self.loss_fn = loss_fn
        self.optim_method = optim_method
        self.lr = lr

    def _batch_loss(self, batch, calculate_metrics=False, prefix="val"):
        input_batch = batch.clone()
        output = self.model(input_batch)

        preds = output["output"]
        hidden = output["hidden_embeddings"]

        loss = self.loss_fn(preds, batch, hidden)

        if calculate_metrics:
            metrics_r = eval_function(preds.detach().cpu(), batch.detach().cpu())
            
            metrics_r["loss"] = loss

            return {f'{prefix}_{k}': v for k, v in metrics_r.items()}
        
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
        optimizer = self.optim_method(self.parameters(), lr=self.lr)
        return optimizer