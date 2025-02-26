# Copyright (c) 2024-present, Royal Bank of Canada.
# Copyright (c) 2019, Carnegie Mellon University.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##################################################################################################
# Code is based on the RFPP (https://arxiv.org/pdf/2405.20320) implementation
# from https://github.com/sangyun884/rfpp by Carnegie Mellon University which is licensed under The BSD 3-Clause Clear License.
# You may obtain a copy of the License at
#
# https://spdx.org/licenses/BSD-3-Clause-Clear.html
#
##################################################################################################

import math
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import argparse
from tqdm import tqdm
import json 
import torch.nn.functional as F
import matplotlib.pyplot as plt
import wandb

# DDP
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from DiT_models import DiT
from tpp_dataset import TPPDataset, data_config_dict
from train_vae.model import Model_VAE

from test_next_event import test_next_event
from test_otd import long_horizon_pred

torch.manual_seed(0)

def ddp_setup(local_rank, num_nodes, num_gpus_per_node, node_rank, master_addr, port):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
        port: Port number to use for initialization
    """
    print(f"Master address is {master_addr}")
    os.environ["MASTER_PORT"] = str(port)
    os.environ["MASTER_ADDR"] = master_addr
    rank = local_rank + num_gpus_per_node * node_rank
    world_size = num_nodes * num_gpus_per_node
    # Windows
    # init_process_group(backend="gloo", rank=rank, world_size=world_size)
    # Linux
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    print(f"Initialized on port {port}")

def get_args():
    parser = argparse.ArgumentParser(description='Configs')
    parser.add_argument('--gpu', type=str, help='gpu num')
    parser.add_argument('--dataname', type=str, help='Dataset name, [taxi, taobao, amazon, stackoverflow, retweet]')
    parser.add_argument('--d_latent', type=int, help='Latent dimension size [8, 16, 32]')
    parser.add_argument('--max_beta', type=str, help='Max beta [0.01, 0.001]')
    parser.add_argument('--dir', type=str, help='Saving directory name')
    parser.add_argument('--tmpdir', type=str, help='Temporary directory', default=None)
    parser.add_argument('--min_len', type=int, default = 1, help='Minimum length of preceding events')
    parser.add_argument('--validate', action='store_true', help='perform validation (True) or test (False)')
    parser.add_argument('--mask', action='store_true', help='mask proceeding events')
    parser.add_argument('--integration_method', type=str, default = "euler", help="ODE integration method ['euler','rk4']")
    parser.add_argument('--test_type', type=str, default = "next", help="Which test to do ['next','likelihood','otd']")
    parser.add_argument('--K', type=int, default = 1, help="Number of initial noise simulations")
    parser.add_argument('--iterations', type=int, default = 1000000, help='Number of iterations')
    parser.add_argument('--batchsize', type=int, default = 4, help='Batch size')
    parser.add_argument('--effective_batchsize', type=int, default = None, help='Effective batch size. If None, same as batchsize. If larger than batchsize, gradient accumulation is used')
    parser.add_argument('--learning_rate', type=float, default = 3e-5, help='Learning rate')
    parser.add_argument('--resume', type=str, default = None, help='Training state path')
    parser.add_argument('--ckpt', type=str, default = None, help='Model ckpt path')
    parser.add_argument('--no_ema', action='store_true', help='use EMA or not')
    parser.add_argument('--ema_after_steps', type=int, default = 1, help='Apply EMA after steps')
    parser.add_argument('--ema_decay', type=float, default = 0.9999, help='EMA decay rate')
    parser.add_argument('--save_iter', type=int, default = 50000, help='Save iteration')
    parser.add_argument('--optimizer', type=str, default = 'adam', help='adam / adamw')
    parser.add_argument('--warmup_steps', type=int, default = 0, help='Learning rate warmup')
    parser.add_argument('--config_de', type=str, default = None, help='Decoder config path, must be .json file')
    parser.add_argument('--loss_type', type=str, default = 'l2', help='loss type, [l2, huber]')
    parser.add_argument('--schedule', type=str, default = 'async', help='Noise schedule, [async, sync, disjoint]')
    parser.add_argument('--port', type=int, default = 12354, help='Port number')
    parser.add_argument('--num_workers', type=int, default=1, help='number of workers')


    parser.add_argument('--compile', action='store_true', help='Compile the model')
    parser.add_argument('--subset', type=int, default = None, help='Subset of the dataset')

    parser.add_argument('--loss_scaling', type=float, default = 1, help='Loss scaling factor')

    arg = parser.parse_args()

    arg.use_ema = not arg.no_ema
    return arg

def get_val_loader(arg, world_size, rank):
    pt_file = "pt_dataset/" + arg.dataname + "/valid"
    val_dataset = TPPDataset(pt_file)
    if arg.subset is not None:
        val_dataset = torch.utils.data.Subset(val_dataset, np.arange(arg.subset))
    # Print len
    if rank == 0:
        print(f"len(val_dataset) = {len(val_dataset)}")

    data_loader = DataLoader(dataset=val_dataset,
                            batch_size=arg.batchsize,
                            shuffle=False,
                            num_workers=arg.num_workers,
                            )
   
    return data_loader

def get_test_loader(arg, world_size, rank):
    pt_file = "pt_dataset/" + arg.dataname + "/test"
    val_dataset = TPPDataset(pt_file)
    if arg.subset is not None:
        val_dataset = torch.utils.data.Subset(val_dataset, np.arange(arg.subset))
    # Print len
    if rank == 0:
        print(f"len(test_dataset) = {len(val_dataset)}")

    data_loader = DataLoader(dataset=val_dataset,
                            batch_size=arg.batchsize,
                            shuffle=False,
                            num_workers=arg.num_workers
                            )
   
    return data_loader

def get_test_fn(arg):
    if arg.test_type == "next":
        return test_next_event
    if arg.test_type == "otd":
        return long_horizon_pred
    return ValueError("arg.test_type must be in ['next','likelihood','otd']")

def get_vae(arg):
    n_head = 1
    factor = 32
    num_layers = 2

    d_numerical = 1
    categories = [data_config_dict[arg.dataname]["data_spec"]["num_event_types"]]

    vae_ckpt = "train_vae/ckpt/" + arg.dataname + "/" + str(arg.d_latent) + "transformer" + str(arg.max_beta)

    vae = Model_VAE(num_layers, d_numerical, categories, arg.d_latent, n_head = n_head, factor = factor, bias = True, transformer = True)
    vae.load_state_dict(torch.load(vae_ckpt + "/model.pt", weights_only=True))
    vae.eval()

    return vae

def main(local_rank: int, num_nodes: int, num_gpus_per_node: int, node_rank: int, master_addr: str, arg):
    port = arg.port

    rank = local_rank + num_gpus_per_node * node_rank
    world_size = num_nodes * num_gpus_per_node

    tags = ["validating" if arg.validate else "testing"]
    if arg.mask:
        tags.append("mask")
    tags.append(arg.test_type)
    tags.append(arg.schedule)
    notes = arg.resume if arg.resume else arg.ckpt
    wandb.init(project="TPP", group=str(arg.d_latent)+arg.dataname+str(arg.max_beta), dir=dir, tags=tags, notes=notes)
    wandb.log({"dataname": arg.dataname})

    ddp_setup(local_rank, num_nodes, num_gpus_per_node, node_rank, master_addr, port)

    device = torch.device(f"cuda:{local_rank}")    

    if arg.validate:
        test_loader = get_val_loader(arg, world_size, rank)
    else:
        test_loader = get_test_loader(arg, world_size, rank)

    vae = get_vae(arg).to(device)

    flow_model = DiT(
        num_rows=data_config_dict[arg.dataname]["data_spec"]["max_len"],
        latent_size=arg.d_latent*3,
        hidden_size=1152,
        depth=7,
        num_heads=16,
        mlp_ratio=4,
        learn_sigma=False
    )

    if rank == 0:
        # Print the number of parameters in the model
        pytorch_total_params = sum(p.numel() for p in flow_model.parameters())
        # Convert to M
        pytorch_total_params = pytorch_total_params / 1000000
        print(f"Total number of the reverse parameters: {pytorch_total_params}M")
        # Save the configuration of flow_model to a json file
        config_dict = {}
        config_dict['num_params'] = pytorch_total_params
        with open(os.path.join(arg.dir, 'config_flow_model.json'), 'w') as f:
            json.dump(config_dict, f, indent = 4)
    
    # Load training state in arg.training_state
    if arg.resume is not None:
        training_state = torch.load(arg.resume, map_location = 'cpu')
        start_iter = training_state['iter']
        flow_model.load_state_dict(training_state['model_state_dict'])
    else:
        start_iter = 0
    if arg.ckpt is not None:
        flow_model.load_state_dict(torch.load(arg.ckpt, map_location = 'cpu'))

    flow_model = flow_model.to(device)

    # DDP
    flow_model = DDP(flow_model, device_ids=[local_rank])
    if arg.compile:
        flow_model = torch.compile(flow_model)# mode="reduce-overhead" raises an error
    
    test_fn = get_test_fn(arg)
    
    test_fn(model=flow_model,valid_dataloader=test_loader,model_vae=vae,device=device,max_len=data_config_dict[arg.dataname]["data_spec"]["max_len"],arg=arg,min_len_const=arg.min_len)
    
    destroy_process_group()

if __name__ == "__main__":
    arg = get_args()

    device_ids = arg.gpu.split(',')
    device_ids = [int(i) for i in device_ids]

    print("torch.cuda.is_available():",torch.cuda.is_available())

    # Process environment variables
    num_nodes = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    num_gpus_per_node = len(device_ids)
    node_rank = int(os.environ['NODE_RANK']) if 'NODE_RANK' in os.environ else 0
    master_addr = os.environ['MASTER_ADDR'] if 'MASTER_ADDR' in os.environ else '127.0.0.1'

    if node_rank == 0:
        if not os.path.exists(arg.dir):
            os.makedirs(arg.dir)
    os.environ["CUDA_VISIBLE_DEVICES"] = arg.gpu
    if arg.tmpdir is None:
        arg.tmpdir = os.path.join(arg.dir, "tmp")

    # Create tmp directory for torch.compile
    if not os.path.exists(arg.tmpdir):
        os.makedirs(arg.tmpdir)
    os.environ['TMPDIR'] = arg.tmpdir

    # world_size = len(device_ids)
    if node_rank == 0:
        with open(os.path.join(arg.dir, "config.json"), "w") as json_file:
            json.dump(vars(arg), json_file, indent = 4)
    # Gradient accumulation
    if arg.effective_batchsize is None:
        arg.effective_batchsize = arg.batchsize
    else:
        assert arg.effective_batchsize >= arg.batchsize
        assert arg.effective_batchsize % arg.batchsize == 0

    log = f"num_nodes: {num_nodes}, num_gpus_per_node: {num_gpus_per_node}, node_rank: {node_rank}, master_addr: {master_addr}, batchsize: {arg.batchsize}, effective_batchsize: {arg.effective_batchsize}, port: {arg.port}"
    print(log)
    if node_rank == 0:
        with open(os.path.join(arg.dir, "log.txt"), "a") as f:
            f.write(log + "\n")

    # DDP
    arg.batchsize = arg.batchsize // num_nodes // num_gpus_per_node
    arg.effective_batchsize = arg.effective_batchsize // num_nodes // num_gpus_per_node
    try:
        main(node_rank, num_nodes, num_gpus_per_node, node_rank, master_addr, arg)
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        destroy_process_group()
        exit(0)