import argparse
import datetime 
import os 
import time 
from torchvision import transforms, datasets
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from DRM import DiffusionRobustModel
from torchvision.utils import save_image
from torch.cuda.amp.autocast_mode import autocast
from load_dataset import LoadDataset, get_subset_random_sampler
# from runtime_args import args


CIFAR10_DATA_DIR = "data/cifar10"
def clean_test(clean_dataset_path, standalone_model, args, sample_output_imgs_folder):
     # get model specific transforms (normalization, resize)

    test_dataset = datasets.CIFAR10(CIFAR10_DATA_DIR, train=False, download=True, transform=transforms.ToTensor())
    test_subset_sampler = get_subset_random_sampler(test_dataset, 0.05)
    loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1,
                                    pin_memory=True, sampler=test_subset_sampler)

    print("Testing standalone on clean data", flush=True)
    image_num = 0
    # Standalone testing on clean data:
    total = 0
    standalone_correct = 0
    progress = tqdm(loader, "CLEANVAL", total=len(loader))
    with torch.no_grad():
        with autocast():
            for x, y in progress:
                x, y = x.cuda(), y.cuda()

                imgs = torch.nn.functional.interpolate(x, (224, 224), mode='bilinear', antialias=True)

                standalone_output = standalone_model(imgs)
                _, standalone_predicted = torch.max(standalone_output[1].data, 1)
                standalone_correct += (standalone_predicted == y).sum().item()
                total += y.size(0) 

                if image_num < 15:
                    for idx, img in enumerate(x):
                        filename = sample_output_imgs_folder + f"/pgd_image_{image_num}.png"
                        save_image(img, filename)
                        break
                    image_num += 1

    return standalone_correct, total

def main(args):
    if args.sigma == 0 and args.epsilon == 0.0:
        print("Base Clean Accuracy, Base PGD Accuracy, DDS PGD Accuracy, Sigma, Epsilon")
    filename = f"{args.ptfile}"
    sample_output_imgs_folder = "samples_" + args.data_folder.split("/")[-1]
    if not os.path.exists(sample_output_imgs_folder) : os.mkdir(sample_output_imgs_folder)
    model = DiffusionRobustModel(filename, sample_output_imgs_folder)
    standalone_model = torch.load(filename)
    
    DATASET_SIZE = 0.1
    IMG_SIZE = 224

    test_dataset = LoadDataset(dataset_folder_path=args.data_folder, image_size=IMG_SIZE, image_depth=3, train=False,
                            transform=transforms.ToTensor(), validate=True)
    loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1,
                                    pin_memory=True)

    # Get the timestep t corresponding to noise level sigma
    target_sigma = args.sigma * 2
    real_sigma = 0
    t = 0
    while real_sigma < target_sigma:
        t += 1
        a = model.diffusion.sqrt_alphas_cumprod[t]
        b = model.diffusion.sqrt_one_minus_alphas_cumprod[t]
        real_sigma = b / a

    #Define the smoothed classifier 
    total = 0
    d_correct = 0
    p_correct = 0
    with torch.no_grad():
        with autocast():
            for i, sample in tqdm(enumerate(loader), total=len(loader)):
                x, y = sample['image'].cuda(non_blocking=True), sample['label'].cuda(non_blocking=True)

                d_out, p_out = model(x, t, y)

                total += y.size(0)
                _, d_predicted = torch.max(d_out[1].data, 1)
                d_correct += (d_predicted == y).sum().item()

                _, p_predicted = torch.max(p_out[1].data, 1)
                p_correct += (p_predicted == y).sum().item()
                del d_out
                del p_out


    clean_dataset_path = "/nfs/stak/users/morgamat/hpc-share/CS_499/CS_499_Term_Project/ImageNet-Models/val_images"
    standalone_correct, standalone_total = clean_test(clean_dataset_path, standalone_model, args, sample_output_imgs_folder)

    # print("Total:", total, "Standalone total:", standalone_total)

    # print()
    # print(f"{'Clean Standalone Model Accuracy':<30}{'PGD Model Accuracy':<30}{'Diffusion Denoised Model Accuracy':<40}{'Noise Sigma'}", flush=True)
    print(f"{100*standalone_correct/standalone_total},{100*p_correct/total},{100*d_correct/total},{args.sigma},{args.epsilon}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict on many examples')
    parser.add_argument("--sigma", type=float, help="noise hyperparameter")
    parser.add_argument("--batch_size", type=int, default=200, help="batch size")
    parser.add_argument("--outfile", type=str, help="output file")
    parser.add_argument("--ptfile", type=str, help="pre-trained classifier file")
    parser.add_argument('--data_folder', type=str, help='Specify the path to the folder where the data is.', required=True)
    parser.add_argument("--epsilon", type=float)
    args = parser.parse_args()

    main(args)