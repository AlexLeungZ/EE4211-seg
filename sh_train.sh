#!/bin/bash
#SBATCH -J Bird
#SBATCH -o Bird_seg_deeplabv3Plus2.out               
#SBATCH -e error.err
#SBATCH --gres=gpu:1
#SBATCH -w node3

echo "Submitted from:"$SLURM_SUBMIT_DIR" on node:"$SLURM_SUBMIT_HOST
echo "Running on node "$SLURM_JOB_NODELIST 
echo "Allocate Gpu Units:"$CUDA_VISIBLE_DEVICES

source /home/xiaoqiguo2/.bashrc

conda activate torch

cd /home/xiaoqiguo2/EE4211/experiment/Deeplabv3_plus/
python ./train.py
