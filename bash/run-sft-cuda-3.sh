#!/bin/sh
#BSUB -gpu "num=3:mode=exclusive_process"
#BSUB -n 3
#BSUB -q gpu
#BSUB -o run-sft-cuda-3.out
#BSUB -e run-sft-cuda-3.err
#BSUB -J easyllm-sft-cuda-3

cd ..
CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1 python run.py
