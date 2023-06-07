
### How to run through diffusion denoiser:

CIFAR-10
```
python ./cifar10/construct.py --sigma 0.25 --batch_size 200 --outfile output.txt --ptfile 20_epoch_model.pt
```

ImageNet
```
python ./imagenet/construct.py --sigma 0.25 --batch_size 200 --outfile output.txt --ptfile pretrained_coatnet.pt
```