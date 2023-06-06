#!/bin/bash
#SBATCH -J train_san_cnn
#SBATCH -p class
#SBATCH -A cs479-579
#SBATCH --gres=gpu:1    
#SBATCH --mem=32G   
#SBATCH -c 16
#SBATCH -t 1-00:00:00       
#SBATCH --export=ALL
# python3 train.py --data_folder datasets/cifar10png/ --num_classes=10 --img_size=224 --gpus 1 $1
python3 train.py --data_folder datasets/cifar100png/ --num_classes=100 --img_size=224 --gpus 1 $1
