import json
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import os
import torchvision.transforms.functional as F
import timm

from valid import validate
import load_dataset as DatasetLoader
from runtime_args import args
from PGD import generate_adversarial_images, save_all_adversarial_images

    
def save_unnormalized_img(img, filename, data_config):
    inverse_transform = transforms.Normalize(mean=[-m/s for m, s in zip(data_config['mean'], data_config['std'])], std=[1/s for s in data_config['std']])
    inversed_image = inverse_transform(img)
    save_image(inversed_image, filename)


device = torch.device("cuda" if torch.cuda.is_available() and args.device == 'gpu' else 'cpu')
LOCAL_MODEL_PATH = "./pretrained_coatnet.pt"
local_model_exists = os.path.exists(LOCAL_MODEL_PATH)


if __name__ == "__main__":
    print(f"PyTorch device: {device}", flush=True)

    path_to_model = args.pretrained_path or LOCAL_MODEL_PATH
    assert path_to_model[-3:] == '.pt'

    print(path_to_model, flush=True)

    DATASET_LEN = args.PGD_image_count
    
    # Dataset Creation
    dataset = DatasetLoader.LoadDataset(dataset_folder_path=args.data_folder, image_depth=args.img_depth,
                                transform=transforms.ToTensor(), train=False, validate=False)

    sampler = DatasetLoader.get_subset_random_sampler(dataset, DATASET_LEN)

    test_generator = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                    pin_memory=True, sampler=sampler)
    
    model = torch.load(path_to_model)
    model = model.to(device)
    model.eval()

    save_path = f"{args.PGD_save_path}/e_{args.PGD_epsilon}_n_{args.PGD_niter}_s_{args.PGD_stepsize/255}"
    
    print(f"\n== Generating Adversarial Images with epsilon {args.PGD_epsilon}\n")
    generate_adversarial_images(args.PGD_image_count, model, test_generator, args.PGD_niter, args.PGD_epsilon, args.PGD_stepsize/255, save_path)
    
    # print("\n== Saving Images to drive\n")
    # save_all_adversarial_images(f"{args.PGD_save_path}/e_{args.PGD_epsilon}_n_{args.PGD_niter}_s_{args.PGD_stepsize/255}", advImages, labels)

    print("done!")

    #validate(model, advImages)