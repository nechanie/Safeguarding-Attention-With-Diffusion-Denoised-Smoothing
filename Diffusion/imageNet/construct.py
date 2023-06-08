import argparse
import datetime 
import os 
import time 
import timm
from torchvision import transforms, datasets
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from DRM import DiffusionRobustModel, save_unnormalized_img

from load_dataset import LoadDataset, get_subset_random_sampler
# from runtime_args import args

# # Device will determine whether to run the training on GPU or CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMG_SIZE = 224

def clean_test(clean_dataset_path, standalone_model, args, sample_output_imgs_folder):
     # get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(standalone_model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    test_dataset = LoadDataset(dataset_folder_path=clean_dataset_path, image_size=IMG_SIZE, image_depth=3, train=False,
                            transform=transforms, validate=True)
    test_subset_sampler = get_subset_random_sampler(test_dataset, 0.02)
    loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1,
                                    pin_memory=True, sampler=test_subset_sampler)

    print("Testing standalone on clean data", flush=True)
    image_num = 0
    # Standalone testing on clean data:
    total = 0
    standalone_correct = 0
    with torch.no_grad():
        for i, sample in tqdm(enumerate(loader), total=len(loader)):
            x, y = sample['image'].cuda(non_blocking=True), sample['label'].cuda(non_blocking=True)

            imgs = torch.nn.functional.interpolate(x, (384, 384), mode='bilinear', antialias=True)

            standalone_output = standalone_model(imgs)
            _, standalone_predicted = torch.max(standalone_output.data, 1)
            standalone_correct += (standalone_predicted == y).sum().item()
            total += y.size(0) 

            if image_num < 15:
                for idx, img in enumerate(x):
                    filename = sample_output_imgs_folder + f"/pgd_image_{image_num}.png"
                    save_unnormalized_img(img, filename, data_config)
                    break
                image_num += 1

    return standalone_correct, total


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
    
    # get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(standalone_model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    model.data_config = data_config # for the denoiser to unnormalize to save image

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
    d_correct = 0
    p_correct = 0
    with torch.no_grad():
        for i, sample in tqdm(enumerate(loader), total=len(loader)):
            x, y = sample['image'].cuda(non_blocking=True), sample['label'].cuda(non_blocking=True)

            d_out, p_out = model(x, t, y)

            total += y.size(0)
            _, d_predicted = torch.max(d_out.data, 1)
            d_correct += (d_predicted == y).sum().item()

            _, p_predicted = torch.max(p_out.data, 1)
            p_correct += (p_predicted == y).sum().item()


    clean_dataset_path = "/nfs/stak/users/morgamat/hpc-share/CS_499/CS_499_Term_Project/ImageNet-Models/val_images"
    standalone_correct, standalone_total = clean_test(clean_dataset_path, standalone_model, args, sample_output_imgs_folder)

    print("Total:", total, "Standalone total:", standalone_total)

    print()
    print(f"{'Clean Standalone Model Accuracy':<30}{'PGD Model Accuracy':<30}{'Diffusion Denoised Model Accuracy':<40}{'Noise Sigma'}", flush=True)
    print(f"{100*standalone_correct/standalone_total:^25}{100*p_correct/total:^25}{100*d_correct/total:^40}{args.sigma:^20}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict on many examples')
    parser.add_argument("--sigma", type=float, help="noise hyperparameter")
    parser.add_argument("--batch_size", type=int, default=200, help="batch size")
    parser.add_argument("--outfile", type=str, help="output file")
    parser.add_argument("--ptfile", type=str, help="pre-trained classifier file")
    parser.add_argument('--data_folder', type=str, help='Specify the path to the folder where the data is.', required=True)
    args = parser.parse_args()

    main(args)