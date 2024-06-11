from algo_reasoning.src.models.network import EncodeProcessDecode
from algo_reasoning.src.data import CLRSDataset, collate
from algo_reasoning.src.losses.CLRSLoss import CLRSLoss
from algo_reasoning.src.lightning.CLRSTask import CLRSTask



import torch
from torch.optim import Adam
import lightning as L
from torch.utils.data import DataLoader
import argparse
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.profilers import AdvancedProfiler

#TODO: Deal with different number of nodes!!

torch.set_float32_matmul_precision('high')

def list_of_strings(arg):
    return arg.split(',')
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Training Parser Options')
    ap.add_argument('--algorithms', default=["bfs"], type=list_of_strings, help="Algorithms for the model to be trained on.")
    ap.add_argument('--path', default="tmp/CLRS30", type=str, help="Path to the dataset folder")
    ap.add_argument('--batch_size', default=16, type=int, help="Number of samples in each training batch")
    ap.add_argument('--n_epochs', default=100, type=int, help="Number of training epochs")
    ap.add_argument('--n_workers', default=8, type=int, help="Number of Data Loading Workers")
    ap.add_argument('--lr', default=1e-3, type=float, help="Initial Learning Rate for ADAM Optimizer")
    ap.add_argument('--lr_decrease_factor', default=0.1, type=float, help="Factor by which the learning rate is going to be reduced after lr_patience epochs without Evaluation perfomance improvement.")
    ap.add_argument('--lr_patience', default=10, type=int, help="Number of epochs without improvement for the learning rate to de decreased")
    ap.add_argument('--model_name', default="EncodeProcessDecode_PGN", type=str, help="Model's name")
    ap.add_argument('--checkpoint_path', default="checkpoints/", type=str, help="Path for checkpoints folder")
    ap.add_argument('--checkpoint_model', default="", type=str, help="Path for pretrained checkpoint model")
    ap.add_argument("--accelerator", default="cpu", type=str, help="Device for the model to be trained on")
    ap.add_argument("--devices",  default=1, type=str, help="Number of devices used for training")
    args = ap.parse_args()

    path = args.path

    train_dataset = CLRSDataset(args.algorithms, "train", path)
    val_dataset = CLRSDataset(args.algorithms, "val", path)
    test_dataset = CLRSDataset(args.algorithms, "test", path)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers, collate_fn=collate)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.n_workers, collate_fn=collate)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.n_workers, collate_fn=collate)

    model = EncodeProcessDecode(args.algorithms)

    loss_fn = CLRSLoss("l2")

    optim_method=Adam

    lightning_module = CLRSTask(
        model=model,
        loss_fn=loss_fn,
        optim_method=optim_method,
        lr=args.lr
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=args.checkpoint_path+f"/{args.model_name}/",
        save_weights_only=True,
        filename=args.model_name+'-{epoch:02d}-{val_loss:.2f}',
        every_n_epochs=1
    )

    #profiler = AdvancedProfiler(dirpath=f"{args.checkpoint_path}{args.model_name}/", filename="perf_logs")

    trainer = L.Trainer(default_root_dir=args.checkpoint_path, 
                        max_epochs=args.n_epochs, 
                        devices=args.devices, 
                        accelerator=args.accelerator, 
                        callbacks=[checkpoint_callback],
                        )#profiler=profiler)
    trainer.fit(lightning_module, train_dataloader, val_dataloader)

    trainer.test(lightning_module, test_dataloader)


