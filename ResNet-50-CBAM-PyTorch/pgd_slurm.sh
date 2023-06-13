#!/bin/bash
#SBATCH -J valid_cnn
#SBATCH -p class
#SBATCH -A cs479-579
#SBATCH --gres=gpu:1    
#SBATCH --mem=64G   
#SBATCH -c 16
#SBATCH -t 1-00:00:00       
#SBATCH --export=ALL
python3 generate_PGD_and_validate.py --data_folder=/nfs/stak/users/morgamat/hpc-share/CS_499/CS_499_Term_Project/ResNet-50-CBAM-PyTorch/datasets/cifar10png \
    --pretrained_path=/nfs/stak/users/morgamat/hpc-share/CS_499/CS_499_Term_Project/ResNet-50-CBAM-PyTorch/runs/cifar_10_runs/vanilla_resnet/resnet/20_epoch_model.pt \
    --PGD_image_count 1000 --batch_size 4 --PGD_save_path=adv_images_niter_5_vanilla/ --PGD_epsilon $1
