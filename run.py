from algo_reasoning.src.models.network import EncodeProcessDecode
from algo_reasoning.src.data import CLRSDataset, CLRSSampler, collate
from algo_reasoning.src.losses.CLRSLoss import CLRSLoss
from algo_reasoning.src.lightning.CLRSTask import CLRSTask
from algo_reasoning.src.specs import CLRS_30_ALGS

import torch
from torch.optim import Adam
import lightning as L
from torch.utils.data import DataLoader
import argparse
from lightning.pytorch.callbacks import ModelCheckpoint

torch.set_float32_matmul_precision('highest')

def list_of_strings(arg):
    return arg.split(',')

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Training Parser Options')
    ap.add_argument('--algorithms', default=CLRS_30_ALGS, type=list_of_strings, help="Algorithms for the model to be trained on.")
    ap.add_argument('--path', default="tmp/CLRS30", type=str, help="Path to the dataset folder")
    ap.add_argument('--batch_size', default=32, type=int, help="Number of samples in each training batch")
    ap.add_argument('--n_epochs', default=61, type=int, help="Number of training epochs")
    ap.add_argument('--n_workers', default=8, type=int, help="Number of Data Loading Workers")
    ap.add_argument('--lr', default=1e-3, type=float, help="Initial Learning Rate for ADAM Optimizer")
    ap.add_argument('--grad_clip', default=1, type=float, help="Gradient clipping value")
    ap.add_argument('--model_name', default="Generalist_GMPNN", type=str, help="Model's name")
    ap.add_argument('--checkpoint_path', default="checkpoints/", type=str, help="Path for checkpoints folder")
    ap.add_argument('--checkpoint_model', default="", type=str, help="Path for pretrained checkpoint model")
    ap.add_argument("--accelerator", default="gpu", type=str, help="Device for the model to be trained on")
    ap.add_argument("--devices",  default=1, type=str, help="Number of devices used for training")
    ap.add_argument("--processor_pretrained_path", default="", type=str, help="Path for processor's weights folder")
    ap.add_argument("--pretrained_path", default="checkpoints/Generalist_GMPNN/Generalist_GMPNN-epoch=39-val_loss=1.29.ckpt", type=str, help="Path for model's weights folder")
    ap.add_argument("--freeze_processor", default=False, type=bool, help="Whether or not to freeze processor's weights.")
    args = ap.parse_args()

    processor = None
    if args.processor_pretrained_path != "":
        processor = EncodeProcessDecode(CLRS_30_ALGS).processor
        state_dict = torch.load(args.processor_pretrained_path, map_location=torch.device("cpu"))["state_dict"]
        new_state_dict = {}
        for key in state_dict:
            if "processor" in key:
                new_state_dict[key.replace("model.processor.", "")] = state_dict[key]

        processor.load_state_dict(new_state_dict)

    path = args.path

    train_dataset = CLRSDataset(args.algorithms, "train", path)
    val_dataset = CLRSDataset(args.algorithms, "val", path)
    test_dataset = CLRSDataset(args.algorithms, "test", path)

    train_sampler = CLRSSampler(train_dataset, algorithms=args.algorithms, batch_size=args.batch_size)
    val_sampler = CLRSSampler(val_dataset, algorithms=args.algorithms, batch_size=args.batch_size)
    test_sampler = CLRSSampler(test_dataset, algorithms=args.algorithms, batch_size=args.batch_size)

    train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=args.n_workers, persistent_workers=True, collate_fn=collate)
    val_dataloader = DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=args.n_workers, persistent_workers=True, collate_fn=collate)
    test_dataloader = DataLoader(test_dataset, batch_sampler=test_sampler, num_workers=args.n_workers, persistent_workers=True, collate_fn=collate)

    model = EncodeProcessDecode(args.algorithms, freeze_processor=args.freeze_processor, pretrained_processor=processor)

    if args.pretrained_path != "":
        state_dict = torch.load(args.pretrained_path, map_location=torch.device("cpu"))["state_dict"]
        new_state_dict = {}
        for key in state_dict:
            if "model" in key:
                new_state_dict[key.replace("model.", "")] = state_dict[key]

        model.load_state_dict(new_state_dict)

    loss_fn = CLRSLoss()

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

    trainer = L.Trainer(default_root_dir=args.checkpoint_path, 
                        max_epochs=args.n_epochs, 
                        devices=args.devices, 
                        accelerator=args.accelerator, 
                        callbacks=[checkpoint_callback],
                        use_distributed_sampler=False,
                        gradient_clip_val=args.grad_clip,
                        )
    
    trainer.fit(lightning_module, train_dataloader, val_dataloader)

    trainer.test(lightning_module, test_dataloader)


