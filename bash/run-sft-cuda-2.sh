#!/bin/sh
#BSUB -gpu "num=2:mode=exclusive_process"
#BSUB -n 2
#BSUB -q gpu
#BSUB -o run-sft-cuda-2.out
#BSUB -e run-sft-cuda-2.err
#BSUB -J easyllm-sft-cuda-2

cd ..
CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1 python run.py
