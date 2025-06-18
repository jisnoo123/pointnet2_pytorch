#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,7


python3 test_classification.py  \
    --log_dir pointnet2_cls_ssg --batch_size 256
            