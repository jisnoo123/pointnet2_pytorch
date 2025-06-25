#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

# --model_nepochs = 10_epochs, 20_epochs , For best model: --model_nepochs = best

rm -rf log/classification/pointnet2_cls_ssg/inference

python3 test_classification.py  \
    --log_dir pointnet2_cls_ssg --batch_size 1024 --model_nepochs 90_epochs
            