#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,6

python3 train_classification.py --model pointnet2_cls_ssg \
    --log_dir pointnet2_cls_ssg --epoch 100 --batch_size 256