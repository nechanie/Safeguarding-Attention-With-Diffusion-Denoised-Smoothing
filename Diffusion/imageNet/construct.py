import argparse
import datetime 
import os 
import time 
import timm
from torchvision import transforms, datasets
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from DRM import DiffusionRobustModel

from load_dataset import LoadDataset, get_subset_random_sampler
# from runtime_args import args

# # Device will determine whether to run the training on GPU or CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    filename = f"imageNet/{args.ptfile}"
    print(filename)
    print("Dataset folder:", args.data_folder)
    sample_output_imgs_folder = "samples_" + args.data_folder.split("/")[-1]
    print("Saving image smaples to:", sample_output_imgs_folder)
    if not os.path.exists(sample_output_imgs_folder) : os.mkdir(sample_output_imgs_folder)

    model = DiffusionRobustModel(filename, sample_output_imgs_folder)
    standalone_model = torch.load(filename)
    standalone_model = standalone_model.to(device)

    
    DATASET_SIZE = 1.0
    IMG_SIZE = 224
    
    # get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(standalone_model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    model.data_config = data_config

    test_dataset = LoadDataset(dataset_folder_path=args.data_folder, image_size=IMG_SIZE, image_depth=3, train=False,
                            transform=transforms, validate=True)
    test_subset_sampler = get_subset_random_sampler(test_dataset, DATASET_SIZE)
    loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1,
                                    pin_memory=True, sampler=test_subset_sampler)

    # Get the timestep t corresponding to noise level sigma
    target_sigma = args.sigma * 2
    real_sigma = 0
    t = 0
    while real_sigma < target_sigma:
        t += 1
        a = model.diffusion.sqrt_alphas_cumprod[t]
        b = model.diffusion.sqrt_one_minus_alphas_cumprod[t]
        real_sigma = b / a

    # Define the smoothed classifier 
    total = 0
    correct = 0
    standalone_correct = 0
    with torch.no_grad():
        for i, sample in tqdm(enumerate(loader), total=len(loader)):
            x, y = sample['image'].cuda(non_blocking=True), sample['label'].cuda(non_blocking=True)

            output = model(x, t)
            _, predicted = torch.max(output.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

        # Standalone testing:
        for i, sample in tqdm(enumerate(loader), total=len(loader)):
            x, y = sample['image'].cuda(non_blocking=True), sample['label'].cuda(non_blocking=True)

            imgs = torch.nn.functional.interpolate(x, (384, 384), mode='bilinear', antialias=True)

            standalone_output = standalone_model(imgs)
            _, standalone_predicted = torch.max(standalone_output.data, 1)
            standalone_correct += (standalone_predicted == y).sum().item()

    print()
    print(f"{'Standalone Model Accuracy':<30}{'Diffusion Denoised Model Accuracy':<40}{'Noise Sigma'}", flush=True)
    print(f"{100*standalone_correct/total:^25}{100*correct/total:^40}{args.sigma:^20}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict on many examples')
    parser.add_argument("--sigma", type=float, help="noise hyperparameter")
    parser.add_argument("--batch_size", type=int, default=200, help="batch size")
    parser.add_argument("--outfile", type=str, help="output file")
    parser.add_argument("--ptfile", type=str, help="pre-trained classifier file")
    parser.add_argument('--data_folder', type=str, help='Specify the path to the folder where the data is.', required=True)
    args = parser.parse_args()

    main(args)