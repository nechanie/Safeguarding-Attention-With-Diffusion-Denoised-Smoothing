# Torch imports
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

# Util imports
from helper_files import load_dataset as DatasetLoader
from helper_files.runtime_args import args

# Other
import PGD

device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'gpu' else 'cpu')


# TODO: test this func
def generate_adversarial_images(count, model, dataset, niter, epsilon):
    generated_images = []





if __name__ == "__main__":
    # Dataset Creation
    dataset = DatasetLoader.LoadDataset(dataset_folder_path=args.data_folder, image_size=args.img_size, image_depth=args.img_depth, train=True,
                            transform=transforms.ToTensor())
    dataloader = DataLoader(dataset, shuffle=True,
                                    num_workers=args.num_workers, pin_memory=True)
    
    model = torch.load(args.pretrained_path, map_location=device)
    model.eval()

    images = generate_adversarial_images(args.PGD_image_count, model, dataloader, args.PGD_niter, args.PGD_epsilon)
    

