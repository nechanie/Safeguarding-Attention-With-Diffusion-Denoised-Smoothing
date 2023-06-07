import os 
import argparse
import time 
import datetime 
from torchvision import transforms, datasets
from DRM import DiffusionRobustModel
from tqdm import tqdm
import torch
from timm.data import create_transform
import timm
from torch.utils.data import DataLoader
from DRM import DiffusionRobustModel
from datasets import load_dataset

def main(args):
    filename = f"imagenet/{args.ptfile}"
    model = DiffusionRobustModel(filename)
    standalone_model = torch.load(filename)
    # get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(model.classifier)
    transform = create_transform(**data_config, is_training=False)
    # If the dataset is gated/private, make sure you have run huggingface-cli login
    dataset = load_dataset("imagenet-1k", split="val", use_auth_token=True)
    dataset.set_transform(transform)
    
    
    # Get the timestep t corresponding to noise level sigma
    target_sigma = args.sigma * 2
    real_sigma = 0
    t = 0
    while real_sigma < target_sigma:
        t += 1
        a = model.diffusion.sqrt_alphas_cumprod[t]
        b = model.diffusion.sqrt_one_minus_alphas_cumprod[t]
        real_sigma = b / a

    loader = DataLoader(dataset, args.batch_size, shuffle=False)
    total = 0
    correct = 0
    standalone_correct = 0
    with torch.no_grad():
        progress_bar = tqdm(loader, "DDS", len(loader))
        for x, y in progress_bar:
            x, y = x.cuda(), y.cuda()
            output = model(x, t)
            _, predicted = torch.max(output.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
        progress_bar_standalone = tqdm(loader, "Standalone", len(loader))
        for x, y in progress_bar_standalone:
            x, y = x.cuda(), y.cuda()
            imgs = torch.nn.functional.interpolate(imgs, (512, 512), mode='bicubic', antialias=True)
            standalone_output = standalone_model(imgs)
            _, standalone_predicted = torch.max(standalone_output.data, 1)
            standalone_correct += (standalone_predicted == y).sum().item()
    if os.path.isfile(args.outfile):
        f = open(args.outfile, "a")
    else:
        f = open(args.outfile, "w")    
        print(f"{'Standalone Model Accuracy':<30}{'Diffusion Denoised Model Accuracy':<40}{'Noise Sigma'}", file=f, flush=True)
    print(f"{100*standalone_correct/total:^25}{100*correct/total:^40}{args.sigma:^20}", file=f, flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict on many examples')
    parser.add_argument("--sigma", type=float, help="noise hyperparameter")
    parser.add_argument("--batch_size", type=int, default=200, help="batch size")
    parser.add_argument("--outfile", type=str, help="output file")
    parser.add_argument("--ptfile", type=str, help="pre-trained classifier file")
    args = parser.parse_args()

    main(args)