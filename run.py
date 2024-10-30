from algo_reasoning.src.models.network import EncodeProcessDecode
from algo_reasoning.src.sampler import CLRSDataset
from algo_reasoning.src.data import OriginalCLRSDataset, CLRSSampler, collate
from algo_reasoning.src.losses.AlgorithmicReasoningLoss import AlgorithmicReasoningLoss
from algo_reasoning.src.lightning.AlgorithmicReasoningTask import AlgorithmicReasoningTask
from algo_reasoning.src.specs import CLRS_30_ALGS

import os
import torch
from torch.optim import Adam, AdamW
import lightning as L
from torch.utils.data import DataLoader, get_worker_info
import argparse
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint

torch.set_float32_matmul_precision('highest')
# Suppress the warning of the wandb
os.environ['WANDB_CONSOLE'] = 'off'

def list_of_strings(arg):
    return arg.split(',')

def list_of_ints(arg):
    return [int(i) for i in arg.split(',')]

ap = argparse.ArgumentParser(description='Training Parser Options')
ap.add_argument('--algorithms', default=CLRS_30_ALGS, type=list_of_strings, 
                help="Algorithms for the model to be trained on.")
ap.add_argument('--nb_nodes', default='4, 7, 11, 13, 16', type=list_of_ints,
                help="Number of nodes in the graphs")
ap.add_argument('--batch_size', default=32, type=int, 
                help="Number of samples in each training batch")
ap.add_argument('--n_epochs', default=100, type=int,
                help="Number of training epochs")
ap.add_argument('--train_steps', default=100, type=int,
                help='Number of training steps per epoch and algorithm')
ap.add_argument('--val_steps', default=32, type=int,
                help="Number of validation steps per algorithm")
ap.add_argument('--test_steps', default=32, type=int,
                help="Number of test steps per algorithm")
ap.add_argument('--n_workers', default=8, type=int,
                help="Number of Data Loading Workers")
ap.add_argument('--lr', default=1e-3, type=float,
                help="Initial Learning Rate for ADAM Optimizer")
ap.add_argument('--grad_clip', default=1, type=float,
                help="Gradient clipping value")
ap.add_argument('--model_name', default="Generalist", type=str,
                help="Model's name")
ap.add_argument('--checkpoint_path', default="checkpoints/", type=str,
                help="Path for checkpoints folder")
ap.add_argument("--checkpoint_module", default="", type=str,
                help="Path for checkpoint module stored by lightning")
ap.add_argument("--accelerator", default="gpu", type=str,
                help="Device for the model to be trained on")
ap.add_argument("--devices", default=1, type=str,
                help="Number of devices used for training")
ap.add_argument("--pretrained_processor", default="", type=str,
                help="Path for processor's weights folder")
ap.add_argument("--freeze_processor", default=False, type=bool,
                help="Whether or not to freeze processor's weights.")
ap.add_argument("--seed", default=7, type=int,
                help="Seed for the random number generator")
ap.add_argument("--algorithms_args", default="algorithm_args/default.yaml", type=str,
                help="Path for the algorithms' arguments file")
ap.add_argument('--static_dataset_path', default="tmp/CLRS30", type=str,
                help="Path to the dataset folder")


def load_algorithm_args(args_file):
    with open(args_file, 'r') as f:
        args = yaml.safe_load(f)

    return args

def load_pretrained_processor(processor_path):
    if processor_path == "":
        return None
    
    processor = EncodeProcessDecode(CLRS_30_ALGS).processor
    state_dict = torch.load(processor_path, map_location=torch.device("cpu"))["state_dict"]
    processor.load_state_dict(state_dict)

    return processor

if __name__ == '__main__':
    args = ap.parse_args()
    
    nb_nodes = args.nb_nodes
    seed = args.seed

    checkpoint_module = args.checkpoint_module if args.checkpoint_module != "" else None

    processor = load_pretrained_processor(args.pretrained_processor)
    algorithm_args = load_algorithm_args(args.algorithms_args)

    train_dataset = CLRSDataset(args.algorithms, nb_nodes, args.batch_size, args.train_steps, seed=seed, algorithms_args=algorithm_args)
    val_dataset = CLRSDataset(args.algorithms, [max(nb_nodes)], args.batch_size, args.val_steps, seed=seed, algorithms_args=algorithm_args)

    def worker_init_fn(w):
        worker_info = get_worker_info()
        
        dataset = worker_info.dataset
        seed = dataset.seed
        worker_id = worker_info.id

        dataset.reset_generator(worker_id + seed)

    train_dataloader = DataLoader(train_dataset, batch_size=None, num_workers=args.n_workers, persistent_workers=True, worker_init_fn=worker_init_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=None, num_workers=args.n_workers, persistent_workers=True, worker_init_fn=worker_init_fn)

    model = EncodeProcessDecode(args.algorithms, freeze_processor=args.freeze_processor, pretrained_processor=processor)

    loss_fn = AlgorithmicReasoningLoss()

    optim_method=AdamW

    if args.checkpoint_module != "":
        lightning_module = AlgorithmicReasoningTask.load_from_checkpoint(checkpoint_module, model=model, loss_fn=loss_fn)
    else:
        lightning_module = AlgorithmicReasoningTask(
            model=model,
            loss_fn=loss_fn,
            optim_method=optim_method,
            lr=args.lr
        )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=args.checkpoint_path+f"/{args.model_name}/",
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
    
    trainer.fit(lightning_module, train_dataloader, val_dataloader, ckpt_path=checkpoint_module)

    # Test with CLRS Original dataset
    test_dataset = OriginalCLRSDataset(args.algorithms, "test", args.static_dataset_path)
    test_sampler = CLRSSampler(test_dataset, algorithms=args.algorithms, batch_size=args.batch_size)
    test_dataloader = DataLoader(test_dataset, batch_sampler=test_sampler, num_workers=args.n_workers, persistent_workers=True, collate_fn=collate)

    trainer.test(lightning_module, test_dataloader)


