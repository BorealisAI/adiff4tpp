# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
from torchdiffeq import odeint
import wandb

from async_lib import obtain_noise_schedule
from otd_utils import distance_between_event_seq

def otd_batch(true_num, true_cat, pred_num, pred_cat, del_cost, trans_cost, num_types, batch_len, pred_len):
    # Batch size
    n = true_num.shape[0]
    W2_dist = 0

    for i in range(n):
        W2_dist += distance_between_event_seq([true_num[i,batch_len[i]-pred_len:batch_len[i],0],
                                               true_cat[i,batch_len[i]-pred_len:batch_len[i],0]], 
                                              [pred_num[i,batch_len[i]-pred_len:batch_len[i],0],
                                               pred_cat[i,batch_len[i]-pred_len:batch_len[i],0]], 
                                              del_cost, 
                                              trans_cost, 
                                              num_types)[0]
    
    return W2_dist

@torch.no_grad()
def long_horizon_pred(model,valid_dataloader,model_vae,device,max_len,arg,min_len_const=None):
    W2_distance_squared = {5:0,10:0,20:0,30:0}
    total_num = 0
    trans_cost = 1
    del_cost = 1

    for i,batch in enumerate(valid_dataloader):
        # Load data
        batch_num, batch_cat, batch_len = batch # (batch_size, num_rows, 1), (batch_size, num_rows, 1), (batch_size)
        total_num += batch_len.shape[0]

        batch_num = batch_num.to(device)
        batch_cat = batch_cat.to(device)

        A = obtain_noise_schedule(arg)(torch.ones_like(batch_len)*max_len,max_len).to(device)

        # True data
        z_token = model_vae.VAE.Tokenizer(batch_num.view(-1,1), batch_cat.view(-1,1))
        z = model_vae.VAE.get_embedding(z_token).view(batch_num.shape[0],batch_num.shape[1],-1)

        for pred_len in W2_distance_squared.keys():
            # Create a mask
            col_indices = torch.arange(z.shape[1]).unsqueeze(0)
            mask = col_indices < (batch_len - pred_len).unsqueeze(1)

            # Initiate noise
            noise_fixed = torch.rand_like(z)

            # Define the ODE function for solving the reverse flow
            def ode_func(t, x):
                t = t.view(-1,1)
                A_t = A(t)
                A_t_dot = A.derivative(t).unsqueeze(-1)
                # Compute vector field: x_0 - epsilon
                v = model(x,A_t)
                # Fix vector fields for preceding events
                v[mask] = (z-noise_fixed)[mask]
                return A_t_dot*v

            # Sample t, zt
            solution = odeint(ode_func, noise_fixed, A.times, rtol=1e-5, atol=1e-5, method=arg.integration_method)
        
            # Extract the result at t=0
            x_restored = solution[-1]

            # Compute one-step prediction
            pred_token = x_restored
            
            # Decode latent event
            pred_num, pred_cat = model_vae.get_decoding(pred_token.reshape(-1,3,pred_token.shape[-1] // 3))
            pred_num = pred_num.view(batch_num.shape[0],batch_num.shape[1],-1).detach().cpu().numpy()
            pred_cat = pred_cat[0].view(batch_cat.shape[0],batch_cat.shape[1],-1)

            # Params for OTD
            num_types = pred_cat.shape[-1]

            pred_cat = pred_cat.argmax(dim = -1).unsqueeze(-1).detach().cpu().numpy()

            # Compute OTD
            W2_distance_squared[pred_len] += otd_batch(batch_num.detach().cpu().numpy(), 
                                                       batch_cat.detach().cpu().numpy(), 
                                                       pred_num, 
                                                       pred_cat, 
                                                       [del_cost], 
                                                       trans_cost, 
                                                       num_types, 
                                                       batch_len.detach().cpu().numpy(), 
                                                       pred_len)
    
            wandb.log({"OTD_"+str(pred_len): W2_distance_squared[pred_len] / total_num})

    return W2_distance_squared
