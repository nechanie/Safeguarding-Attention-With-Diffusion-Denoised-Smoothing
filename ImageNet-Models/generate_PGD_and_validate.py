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



device = torch.device("cuda" if torch.cuda.is_available() and args.device == 'gpu' else 'cpu')
LOCAL_MODEL_PATH = "./pretrained_coatnet.pt"
local_model_exists = os.path.exists(LOCAL_MODEL_PATH)


if __name__ == "__main__":
    print(f"PyTorch device: {device}")

    print("Downloaded test image", flush=True)


    model = None
    if local_model_exists:
        print("Using local model", flush=True)
        model = torch.load("pretrained_coatnet.pt")
    else:
        print("Fetching remote model", flush=True)
        model = timm.create_model('coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k', pretrained=True)
        torch.save(model, LOCAL_MODEL_PATH)


    path_to_model = args.pretrained_path or LOCAL_MODEL_PATH
    assert path_to_model[-3:] == '.pt'

    print(path_to_model)


    DATASET_SIZE = 0.02
    

    # get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    # Dataset Creation
    dataset = DatasetLoader.LoadDataset(dataset_folder_path=args.data_folder, image_depth=args.img_depth,
                                transform=transforms, validate=True)

    sampler = DatasetLoader.get_subset_random_sampler(dataset, DATASET_SIZE)

    test_generator = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                    pin_memory=True, sampler=sampler)
    
    
    model = torch.load(path_to_model, map_location=device)
    model.eval()
    

    print(f"\n== Generating Adversarial Images with epsilon {args.PGD_epsilon}\n")
    clean, advImages, labels = generate_adversarial_images(args.PGD_image_count, model, test_generator, args.PGD_niter, args.PGD_epsilon, args.PGD_stepsize/255)
    print("\n== Saving Images to drive\n")
    save_all_adversarial_images(f"{args.PGD_save_path}/e_{args.PGD_epsilon}_n_{args.PGD_niter}_s_{args.PGD_stepsize/255}", advImages, labels)

    print("done!")

    #validate(model, advImages)