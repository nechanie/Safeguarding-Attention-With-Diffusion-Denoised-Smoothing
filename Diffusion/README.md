
### How to run through diffusion denoiser:

CIFAR-10
1. Make sure you have `cifar10_uncond_50M_500K.pt` in `/cifar10`
```
python3 ./cifar10/construct.py --sigma 0.25 --batch_size 200 --outfile output.txt --ptfile 20_epoc
h_model.pt --sigma 0.75 --data_folder=/nfs/stak/users/morgamat/hpc-share/CS_499/CS_499_Term_Project/ResNet-50-CBAM-PyTorch/adv_images_niter_5/epsilon_0.0_niter_5
```

ImageNet
1. Make sure you have `256x256_diffusion_uncond.pt` in `/imageNet`
2. Use slurm for more memory:
```
sbatch imageNet_diffusion_slurm.sh
```