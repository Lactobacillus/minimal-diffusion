#!/bin/bash

# CUDA_VISIBLE_DEVICES=2,3,4,5 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8101 main.py \
#     --arch UNet --dataset mnist --class-cond --epochs 100 --batch-size 256 --sampling-steps 100 \
#     --data-dir ~/dataset/mnist

CUDA_VISIBLE_DEVICES=2,3,4,5 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8105  main.py \
    --arch UNet --dataset cifar10 --class-cond --epochs 500 --batch-size 256 --sampling-steps 100 \
    --data-dir ~/dataset/cifar10
