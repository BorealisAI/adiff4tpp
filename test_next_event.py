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

def sample_t(num_samples):
    return torch.rand(num_samples)

@torch.no_grad()
def test_next_event(model,valid_dataloader,model_vae,device,max_len,arg,min_len_const=None):
    mse_loss = 0
    accuracy = 0
    total_num = 0

    for i,batch in enumerate(valid_dataloader):
        # Load data
        batch_num, batch_cat, batch_len = batch

        # Compute maximum number of preceding events
        max_batch_len = torch.max(batch_len).item()

        batch_num = batch_num.to(device)
        batch_cat = batch_cat.to(device)

        A = obtain_noise_schedule(arg)(torch.ones_like(batch_len)*max_len,max_len).to(device)

        # True data
        z_token = model_vae.VAE.Tokenizer(batch_num.view(-1,1), batch_cat.view(-1,1))
        z = model_vae.VAE.get_embedding(z_token).view(batch_num.shape[0],batch_num.shape[1],-1)

        # Initiate mask
        pred_token = torch.zeros_like(z)
        mask_tensor = batch_num != 0.0
        total_num += mask_tensor.sum()
        pred_num = torch.zeros_like(batch_num)
        pred_cat = None
        
        for preceding_len in range(1,max_batch_len):
            # Initiate noise
            noise_fixed = torch.rand_like(z)
            if not (batch_len > preceding_len).any():
                continue  # Skip iteration if no valid predictions

            if arg.mask:
                # Zero out proceeding noise
                noise_fixed[:,preceding_len+1:,:] = 0
                # Initiate causal mask
                causal_mask = torch.ones(1,1,noise_fixed.shape[1],noise_fixed.shape[1], dtype=noise_fixed.dtype, device=noise_fixed.device)
                causal_mask[:,:,:,preceding_len+1:] = 0

            # Define the ODE function for solving the reverse flow
            def ode_func(t, x):
                t = t.view(-1,1)
                A_t = A(t)
                A_t_dot = A.derivative(t).unsqueeze(-1)
                # Compute vector field: x_0 - epsilon
                if arg.mask:
                    A_t[:,preceding_len+1:] = 0
                    v = model(x,A_t,causal_mask)
                else:
                    v = model(x,A_t)
                # Fix vector fields for preceding events
                v[:,:preceding_len,:] = (z-noise_fixed)[:,:preceding_len,:]
                return A_t_dot*v

            if arg.schedule == "sync":
                ### SYNCHRONOUS DIFFUSION
                init_A_t = A(A.times[preceding_len])
                init_cond = (1-torch.zeros_like(init_A_t)).unsqueeze(-1) * noise_fixed
                solution = odeint(ode_func, init_cond, A.times, rtol=1e-5, atol=1e-5, method=arg.integration_method)
            elif arg.schedule == "disjoint":
                ### DISJOINT DIFFUSION ###
                init_A_t = A(A.times[preceding_len])
                init_cond = init_A_t.unsqueeze(-1) * z + (1-init_A_t).unsqueeze(-1) * noise_fixed
                solution = odeint(ode_func, init_cond, A.times[preceding_len:preceding_len+1], rtol=1e-5, atol=1e-5, method=arg.integration_method)
            else:
                ### ASYNCHRONOUS DIFFUSION
                init_A_t = A(A.times[preceding_len])
                init_cond = init_A_t.unsqueeze(-1) * z + (1-init_A_t).unsqueeze(-1) * noise_fixed
                solution = odeint(ode_func, init_cond, A.times[preceding_len:(A.times.shape[0]//2)+preceding_len], rtol=1e-5, atol=1e-5, method=arg.integration_method)

            # Extract the result at t=0
            x_restored = solution[-1]

            # Compute one-step prediction
            pred_token[:,preceding_len,:] = x_restored[:,preceding_len,:]
            wandb.log({"step": mask_tensor[:,:preceding_len].sum()})
        
        # Decode latent event
        one_step_pred_num, one_step_pred_cat = model_vae.get_decoding(pred_token.view(-1,3,pred_token.shape[-1] // 3))
        one_step_pred_num = one_step_pred_num.view(batch_num.shape[0],batch_num.shape[1],-1)
        one_step_pred_cat = one_step_pred_cat[0].view(batch_num.shape[0],batch_num.shape[1],-1).to(device)

        pred_num += one_step_pred_num
        if pred_cat is None:
            pred_cat = torch.zeros_like(one_step_pred_cat)
        pred_cat += one_step_pred_cat
        
        pred_cat = pred_cat.argmax(dim = -1).unsqueeze(-1)

        # True event
        true_num, true_cat = batch_num, batch_cat

        mask_tensor = mask_tensor.to(device)

        # Evaluate
        for i in range(max_batch_len):
            wandb.log({"mse_validation": torch.sqrt(torch.sum(((pred_num[:,i:] - true_num[:,i:]) * mask_tensor[:,i:]) ** 2) / torch.sum(mask_tensor[i:])).item()})
            wandb.log({"accuracy_validation": (torch.sum((true_cat[:,i:] == pred_cat[:,i:]) * mask_tensor[:,i:]) / torch.sum(mask_tensor[:,i:])).item()})

        mse_loss += torch.sum(((one_step_pred_num - true_num) * mask_tensor) ** 2)
        accuracy += torch.sum((true_cat == pred_cat) * mask_tensor)

        # Logging
        wandb.log({"mse_validation": torch.sqrt(mse_loss/total_num).item(), "step": total_num})
        wandb.log({"accuracy_validation": (accuracy/total_num).item(), "step": total_num})
    return torch.sqrt(mse_loss/total_num), accuracy/total_num
