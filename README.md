# ADiff4TPP: Asynchronous Diffusion Models for Temporal Point Processes

![image](./assets/figure2.png)

## Official Implementation

This repository contains the implementation of the paper:

ADiff4TPP: Asynchronous Diffusion Models for Temporal Point Processes

Amartya Mukherjee, Ruizhi Deng, He Zhao, Yuzhen Mao, Leonid Sigal, Frederick Tung, 2025.

## Instructions

### Environment Setup

To set up the environment, run the following commands.
```
conda env create -f environment.yml
conda activate adiff4tpp
```

### Dataset and Preprocessing

The datasets are downloaded from the following [Google Drive link](https://drive.google.com/drive/u/0/folders/1f8k82-NL6KFKuNMsUwozmbzDSFycYvz7). All the datasets are provided by the authors of [EasyTPP (Xue et al., 2024)](https://github.com/ant-research/EasyTemporalPointProcess/tree/main).

For preprocessing, please run ```python tpp_dataset.py``` to save the datasets as torch tensors.

### Training

#### Variational Auto Encoder

```
cd train_vae
python main.py --dataname=[dataname] --gpu=0 --d_latent=32 --max_beta=0.01 --num_epochs=1000
```

#### Diffusion Model

```
python train.py --gpu=0,1,2,3 --dataname=[dataname] --d_latent=32 --max_beta=0.01 --dir=[model_dir] --port=12345 --batchsize=4 --mask
```

### Testing

```
python test.py --gpu=0 --dataname=[dataname] --d_latent=32 --max_beta=0.01 --dir=[log_dir]  --port=12345 --batchsize=2000 --ckpt=[model_dir] --integration_method=rk4 --test_type=[next/otd]
```

`[dataname]` can be one of `[taxi,taobao,amazon,retweet,stackoverflow]`.

`[model_dir]` is the directory where you plan to store the model.

`[log_dir]` is where you plan to store `log.txt` and `wandb` files.

`test_type` is `next` if you want to perform next event prediction and `otd` if you plan to perform long horizon prediction.


## License

The source code is licensed under the CC BY-NC-SA 4.0 license.

## Acknowledgement

This code was built on top of the following repositories:

[Elucidating the Design Space of Diffusion-Based Generative Models (EDM) (Karras et al., 2022)](https://github.com/NVlabs/edm)

[Improving the Training of Rectified Flows (2-Rectified Flow++) (Lee at al., 2024)](https://github.com/sangyun884/rfpp)

[Scalable Diffusion Models with Transformers (DiT) (Peebles and Xie, 2023)](https://github.com/facebookresearch/DiT)

[Mixed-Type Tabular Data Synthesis with Score-based Diffusion in Latent Space (Tabsyn) (Zhang et al., 2024)](https://github.com/amazon-science/tabsyn)

[Neural Hawkes Particle Smoothing (Mei et al., 2019)](https://github.com/hongyuanmei/neural-hawkes-particle-smoothing)