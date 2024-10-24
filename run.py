from algo_reasoning.src.models.network import EncodeProcessDecode
from algo_reasoning.src.sampler import CLRSDataset
from algo_reasoning.src.losses.CLRSLoss import CLRSLoss
from algo_reasoning.src.lightning.CLRSTask import CLRSTask
from algo_reasoning.src.specs import CLRS_30_ALGS

import os
import torch
from torch.optim import Adam
import lightning as L
from torch.utils.data import DataLoader, get_worker_info
import argparse
from lightning.pytorch.callbacks import ModelCheckpoint

torch.set_float32_matmul_precision('highest')
# Suppress the warning of the wandb
os.environ['WANDB_CONSOLE'] = 'off'

def list_of_strings(arg):
    return arg.split(',')

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Training Parser Options')
    ap.add_argument('--algorithms', default=CLRS_30_ALGS, type=list_of_strings, help="Algorithms for the model to be trained on.")
    ap.add_argument('--path', default="tmp/CLRS30", type=str, help="Path to the dataset folder")
    ap.add_argument('--batch_size', default=32, type=int, help="Number of samples in each training batch")
    ap.add_argument('--n_epochs', default=100, type=int, help="Number of training epochs")
    ap.add_argument('--n_workers', default=8, type=int, help="Number of Data Loading Workers")
    ap.add_argument('--lr', default=1e-3, type=float, help="Initial Learning Rate for ADAM Optimizer")
    ap.add_argument('--grad_clip', default=1, type=float, help="Gradient clipping value")
    ap.add_argument('--model_name', default="Generalist", type=str, help="Model's name")
    ap.add_argument('--checkpoint_path', default="checkpoints/", type=str, help="Path for checkpoints folder")
    ap.add_argument('--checkpoint_model', default="", type=str, help="Path for pretrained checkpoint model")
    ap.add_argument("--accelerator", default="gpu", type=str, help="Device for the model to be trained on")
    ap.add_argument("--devices",  default=1, type=str, help="Number of devices used for training")
    ap.add_argument("--processor_pretrained_path", default="", type=str, help="Path for processor's weights folder")
    ap.add_argument("--pretrained_path", default="", type=str, help="Path for model's weights folder")
    ap.add_argument("--freeze_processor", default=False, type=bool, help="Whether or not to freeze processor's weights.")
    args = ap.parse_args()

    nb_nodes = [4, 7, 11, 13, 16]
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

    algorithms_args = {}
    # Default arguments for algorithms
    p = tuple([0.1 + 0.1 * i for i in range(9)])
    graph_algos = ["dfs", "bfs", "topological_sort", "articulation_points", "bridges", "strongly_connected_components", "mst_kruskal", "mst_prim", "bellman_ford", "dijkstra", "dag_shortest_paths", "floyd_warshall"]
    for _algo in graph_algos:
        if _algo in ['articulation_points', 'bridges', 'mst_kruskal']:
            p = tuple((torch.tensor(p) / 2).tolist())
        algorithms_args[_algo] = p 

    train_dataset = CLRSDataset(args.algorithms, nb_nodes, args.batch_size, 1000, seed=7, algorithms_args=algorithms_args)
    val_dataset = CLRSDataset(args.algorithms, [max(nb_nodes)], args.batch_size, 32, seed=7, algorithms_args=algorithms_args)
    test_dataset = CLRSDataset(args.algorithms, [64], args.batch_size, 32, seed=7, algorithms_args=algorithms_args)

    def worker_init_fn(w):
        worker_info = get_worker_info()
        
        dataset = worker_info.dataset
        seed = dataset.seed
        worker_id = worker_info.id

        dataset.reset_generator(worker_id + seed)

    train_dataloader = DataLoader(train_dataset, batch_size=None, num_workers=args.n_workers, persistent_workers=True, worker_init_fn=worker_init_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=None, num_workers=args.n_workers, persistent_workers=True, worker_init_fn=worker_init_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=None, num_workers=args.n_workers, persistent_workers=True, worker_init_fn=worker_init_fn)

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


