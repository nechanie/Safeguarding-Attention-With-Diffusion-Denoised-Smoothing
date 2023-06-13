#!/bin/bash
#SBATCH -J valid_cnn
#SBATCH -p class
#SBATCH -A cs479-579
#SBATCH --gres=gpu:1    
#SBATCH --mem=64G
#SBATCH -c 16
#SBATCH -t 1-00:00:00       
#SBATCH --export=ALL
python cifar10/construct.py --sigma $1 --batch_size 1 --ptfile /nfs/stak/users/morgamat/hpc-share/CS_499/CS_499_Term_Project/ResNet-50-CBAM-PyTorch/runs/cifar_10_runs/vanilla_resnet/resnet/20_epoch_model.pt \
    --data_folder /nfs/stak/users/morgamat/hpc-share/CS_499/CS_499_Term_Project/ResNet-50-CBAM-PyTorch/adv_images_niter_5_vanilla/e_$2_n_5_s_0.00784313725490196 --epsilon $2 >> output.csv

# python3 ./cifar10/construct.py --sigma 0.25 --batch_size 1 --outfile output.txt --ptfile /nfs/stak/users/morgamat/hpc-share/CS_499/CS_499_Term_Project/ResNet-50-CBAM-PyTorch/runs/cifar_10_runs/vanilla_resnet/resnet/20_epoch_model.pt \
#     --sigma 0.75 --data_folder=/nfs/stak/users/morgamat/hpc-share/CS_499/CS_499_Term_Project/ResNet-50-CBAM-PyTorch/adv_images_niter_5_vanilla/e_0.005_n_5_s_0.00784313725490196


# Note: our original (presented) results are for resnet_cbam
# Now we need to train vanilla_resnet