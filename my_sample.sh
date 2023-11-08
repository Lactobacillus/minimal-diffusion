#!/bin/bash

sampling_args="--arch UNet --class-cond --sampling-steps 250 --sampling-only --save-dir ./sampled_images/"

# CUDA_VISIBLE_DEVICES=2,3,4,5 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8101 main.py \
#     --arch UNet --dataset mnist --batch-size 256 --num-sampled-images 50000 $sampling_args \
#     --pretrained-ckpt ./trained_models/UNet_mnist-epoch_100-timesteps_1000-class_condn_True_ema_0.9995.pt

CUDA_VISIBLE_DEVICES=2,3,4,5 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8103 main.py \
    --arch UNet --dataset cifar10 --batch-size 256 --num-sampled-images 50000 $sampling_args \
    --pretrained-ckpt ./trained_models/UNet_cifar10-epoch_500-timesteps_1000-class_condn_True_ema_0.9995.pt 
