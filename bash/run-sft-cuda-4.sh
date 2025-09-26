#!/bin/sh
#BSUB -gpu "num=4:mode=exclusive_process"
#BSUB -n 4
#BSUB -q gpu
#BSUB -o run-sft-cuda-4.out
#BSUB -e run-sft-cuda-4.err
#BSUB -J easyllm-sft-cuda-4

cd ..
CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1 python run.py
