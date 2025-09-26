#!/bin/sh
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 1
#BSUB -q gpu
#BSUB -o run-sft-cuda-1.out
#BSUB -e run-sft-cuda-1.err
#BSUB -J easyllm-sft-cuda-1

cd ..
python run.py
