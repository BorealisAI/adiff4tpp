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

from EMA import EMA
from async_lib import obtain_noise_schedule
from DiT_models import DiT
from torch_utils.misc import InfiniteSampler
from fp16_utils import DynamicLossScaler
from tpp_dataset import TPPDataset, data_config_dict
from train_vae.model import Model_VAE

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
    parser.add_argument('--mask', action='store_true', help='mask proceeding events')
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


def train_rectified_flow(rank, arg, model, optimizer, data_loader, valid_loader, iterations, device, start_iter, warmup_steps, dir, learning_rate, 
                         ema_after_steps, use_ema, world_size, save_iter, model_vae, max_len):
    if rank == 0:
        wandb.init(project="TPP", group=str(arg.d_latent)+arg.dataname+str(arg.max_beta), dir=dir, tags=["training"])
        wandb.log({"dataname": arg.dataname})
    # use tqdm if rank == 0

    gradient_accumulation_steps = arg.effective_batchsize // arg.batchsize
    if rank == 0:
        log = f"gradient_accumulation_steps: {gradient_accumulation_steps}"
        print(log)
        with open(os.path.join(dir, "log.txt"), "a") as f:
            f.write(log + "\n")
    i_effective = start_iter
    cnt = 0 # Count the number of backward() calls
    iterations_effective = (iterations - start_iter) * gradient_accumulation_steps # Total number of backward() calls
    iterations_effective += 1000 # Since we sometimes skip the update, safely add some extra iterations

    noise_fixed = None # For visualization
    label_fixed_onehot = None # For visualization

    tqdm_ = tqdm if rank == 0 else lambda x: x

    # Define loss function
    if arg.loss_type == 'l2-squared':
        def loss_func(x, y, A_prime):
            return torch.mean(A_prime**2*(x - y)**2)
    elif arg.loss_type == 'l2':
        def loss_func(x, y, A_prime):
            return torch.sqrt(torch.mean(A_prime**2*(x - y)**2))
    elif arg.loss_type == 'huber':
        def loss_func(x, y, A_prime):
            data_dim = x.shape[1] * x.shape[2] * x.shape[3]
            huber_c = 0.00054 * math.sqrt(data_dim)
            loss = torch.sum(A_prime**2*(x - y)**2)
            loss = torch.sqrt(loss + huber_c**2) - huber_c
            return loss / data_dim
    else:
        raise NotImplementedError(f"Loss type {arg.loss_type} not implemented")

    train_iter = iter(data_loader)
    val_iter = iter(valid_loader)
    optimizer.zero_grad()

    loss_scaler = DynamicLossScaler(init_scale=arg.loss_scaling, scale_window = 10000)

    for cnt in tqdm_(range(iterations_effective)):
        if use_ema and i_effective > ema_after_steps:
            optimizer.ema_start()
        # Learning rate warmup
        if i_effective < warmup_steps:
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate * np.minimum(i_effective / warmup_steps, 1)
        
        # Load data
        batch_num, batch_cat, batch_len = next(train_iter)

        batch_num = batch_num.to(device)
        batch_cat = batch_cat.to(device)

        A = obtain_noise_schedule(arg)(batch_len,max_len).to(device)
        attn_mask = A.attn_mask

        z_token = model_vae.VAE.Tokenizer(batch_num.view(-1,1), batch_cat.view(-1,1))
        z = model_vae.VAE.get_embedding(z_token).view(batch_num.shape[0],batch_num.shape[1],3*arg.d_latent)

        noise_fixed = torch.randn_like(z)

        # Sample t, zt
        t = sample_t(z.shape[0]).view(-1,1).to(device) # (batchsize,)
        A_t = A(t)
        A_t_dot = A.derivative(t).unsqueeze(-1)
        zt = A_t.unsqueeze(-1)*z+(1-A_t.unsqueeze(-1))*noise_fixed
        target = z - noise_fixed
        
        # Forward pass
        if arg.mask:
            pred = model(zt, A_t, attn_mask)
        else:
            pred = model(zt, A_t)

        # Compute loss
        loss_dict = {}
        loss = loss_func(pred, target, A_t_dot)
        loss_dict[arg.loss_type] = loss.mean().item()

        loss = loss.mean()

        # Loss scaling for mixed precision training
        if arg.loss_scaling == 1:
            loss_scale = 1
        else:
            loss_scale = loss_scaler.loss_scale

        (loss * loss_scale / gradient_accumulation_steps).backward()
        cnt += 1

        has_overflow = loss_scaler.has_overflow(model.parameters())
        loss_scaler.update_scale(has_overflow)


        if cnt % gradient_accumulation_steps == 0:
            if not has_overflow:
                if loss_scale != 1:
                    for param in model.parameters():
                        param.grad.data *= 1 / loss_scale
                optimizer.step()
                optimizer.zero_grad()
                i_effective += 1
            else:
                optimizer.zero_grad()

        else:
            if has_overflow:
                optimizer.zero_grad()
            continue # Skip logging, visualization, and saving

        ########### Logging, visualization, and saving ###########
        
        if i_effective % 1000 == 1 and rank == 0:
            log = f"Iteration {i_effective}: lr {optimizer.param_groups[0]['lr']} "
            for key in loss_dict:
                log += f"{key} {loss_dict[key]:.8f} "
            log += f"loss_scale {loss_scale:.8f}"
            log += "\n"
            print(log)
            # Log scalars
            wandb.log({"lr": optimizer.param_groups[0]['lr'], "step": i_effective})
            wandb.log({"loss": loss.item(), "step": i_effective})
            wandb.log({"loss_scale": loss_scale, "step": i_effective})
            # Log dictionary items
            for key in loss_dict:
                wandb.log({key: loss_dict[key], "step": i_effective})
            # Log to .txt file
            with open(os.path.join(dir, 'log.txt'), 'a') as f:
                f.write(log)

        if i_effective % save_iter == 0 and rank == 0:
            if use_ema:
                optimizer.swap_parameters_with_ema(store_params_in_ema=True)
                torch.save(model.module.state_dict(), os.path.join(dir, f"flow_model_{i_effective}_ema.pth"))
                optimizer.swap_parameters_with_ema(store_params_in_ema=True)
            else:
                torch.save(model.module.state_dict(), os.path.join(dir, f"flow_model_{i_effective}.pth"))

            # Save training state
            d = {}
            d['optimizer_state_dict'] = optimizer.state_dict()
            d['model_state_dict'] = model.module.state_dict()
            d['iter'] = i_effective
            # save
            torch.save(d, os.path.join(dir, f"training_state_{i_effective}.pth"))  
        if i_effective % 5000 == 0 and rank == 0 and i_effective > 0:
            # Save the latest training state
            d = {}
            d['optimizer_state_dict'] = optimizer.state_dict()
            d['model_state_dict'] = model.module.state_dict()
            d['iter'] = i_effective
            # save
            torch.save(d, os.path.join(dir, f"training_state_latest.pth"))  

    return

def get_loader(arg, world_size, rank):
    pt_file = "pt_dataset/" + arg.dataname + "/train"
    train_dataset = TPPDataset(pt_file)
    if arg.subset is not None:
        train_dataset = torch.utils.data.Subset(train_dataset, np.arange(arg.subset))
    # Print len
    if rank == 0:
        print(f"len(train_dataset) = {len(train_dataset)}")

    data_loader = DataLoader(dataset=train_dataset,
                            batch_size=arg.batchsize,
                            shuffle=False,
                            drop_last=True,
                            num_workers=arg.num_workers,
                            pin_memory=True,
                            sampler = InfiniteSampler(train_dataset, num_replicas=world_size, rank=rank)
                            )
   
    return data_loader

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
                            drop_last=True,
                            num_workers=arg.num_workers,
                            pin_memory=True,
                            sampler = InfiniteSampler(val_dataset, num_replicas=world_size, rank=rank)
                            )
   
    return data_loader

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

def sample_t(num_samples):
    return torch.rand(num_samples)

def main(local_rank: int, num_nodes: int, num_gpus_per_node: int, node_rank: int, master_addr: str, arg):
    port = arg.port

    rank = local_rank + num_gpus_per_node * node_rank
    world_size = num_nodes * num_gpus_per_node

    ddp_setup(local_rank, num_nodes, num_gpus_per_node, node_rank, master_addr, port)

    device = torch.device(f"cuda:{local_rank}")    

    data_loader = get_loader(arg, world_size, rank)

    valid_loader = get_val_loader(arg, world_size, rank)

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

    
    optimizer = torch.optim.Adam(flow_model.parameters(), lr=arg.learning_rate, betas = (0.9, 0.999), eps=1e-8)

    if arg.use_ema:
        optimizer = EMA(optimizer, ema_decay=arg.ema_decay)

    if arg.resume is not None:
        optimizer.load_state_dict(training_state['optimizer_state_dict'])
        print(f"Loaded training state from {arg.resume} at iter {start_iter}")
        del training_state

    # DDP
    flow_model = DDP(flow_model, device_ids=[local_rank])
    if arg.compile:
        flow_model = torch.compile(flow_model)# mode="reduce-overhead" raises an error
    
    train_rectified_flow(rank = rank, arg = arg, model = flow_model, optimizer = optimizer,
                        data_loader = data_loader, valid_loader = valid_loader, iterations = arg.iterations, device = device, start_iter = start_iter,
                        warmup_steps = arg.warmup_steps, dir = arg.dir, learning_rate = arg.learning_rate,
                        use_ema = arg.use_ema, ema_after_steps = arg.ema_after_steps, world_size=world_size,
                        save_iter = arg.save_iter, model_vae = vae, max_len = data_config_dict[arg.dataname]["data_spec"]["max_len"])
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
       mp.spawn(main, args=(num_nodes, num_gpus_per_node, node_rank, master_addr, arg), nprocs=num_gpus_per_node)
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        destroy_process_group()
        exit(0)
