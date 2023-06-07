#!/bin/bash
#SBATCH -J valid_cnn
#SBATCH -p class
#SBATCH -A cs479-579
#SBATCH --gres=gpu:1    
#SBATCH --mem=32G   
#SBATCH -c 16
#SBATCH -t 1-00:00:00       
#SBATCH --export=ALL
python3 ./cifar10/construct.py --sigma 0.25 --batch_size 1 --outfile output.txt --ptfile 20_epoch_model.pt --sigma 0.75 --data_folder=/nfs/stak/users/morgamat/hpc-share/CS_499/CS_499_Term_Project/ResNet-50-CBAM-PyTorch/adv_images_niter_5/epsilon_0.0_niter_5