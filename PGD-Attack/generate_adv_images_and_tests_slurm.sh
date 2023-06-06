#!/bin/bash
#SBATCH -J train_san_cnn
#SBATCH -p class
#SBATCH -A cs479-579
#SBATCH --gres=gpu:1    
#SBATCH --mem=32G   
#SBATCH -c 16
#SBATCH -t 1-00:00:00       
#SBATCH --export=ALL


python3 create_adversarial_images.py  \
    --data_folder ../datasets/intel_dataset \
    --pretrained_path ../ResNet-50-CBAM-PyTorch/pretrained_weights/intel_dataset_clean_models/2023-5-13__14-40-1/resnet/30_epoch_model.pt \
    --batch_size 20