#!/bin/bash
#SBATCH -J valid_cnn
#SBATCH -p class
#SBATCH -A cs479-579
#SBATCH --gres=gpu:1    
#SBATCH --mem=64G   
#SBATCH -c 16
#SBATCH -t 1-00:00:00       
#SBATCH --export=ALL
python3 generate_PGD_and_validate.py --data_folder=/nfs/stak/users/morgamat/hpc-share/CS_499/CS_499_Term_Project/ImageNet-Models/val_images --PGD_image_count 1000 --batch_size 4 --PGD_epsilon $1
