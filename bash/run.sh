#!/bin/sh
#BSUB -gpu "num=1"
#BSUB -n 1
#BSUB -m gpu01
#BSUB -o run.out
#BSUB -e run.err
#BSUB -J easyllm

cd ..
python run.py