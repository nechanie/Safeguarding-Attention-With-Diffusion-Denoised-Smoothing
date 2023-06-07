import argparse
import matplotlib.pyplot as plt
import numpy as np
import time
import timm
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

from runtime_args import args
from load_dataset import LoadDataset, get_subset_random_sampler
from plot import plot_loss_acc
from helpers import calculate_accuracy

"""Python script for evaluating a pretrained model."""

# Device will determine whether to run the training on GPU or CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def validate(model, data_loader) -> float:
    # In test phase, we don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        total_accuracy = 0
        total_batches = 0
        for i, sample in tqdm(enumerate(data_loader), total=len(data_loader)):
            batch_x, batch_y = sample['image'].cuda(non_blocking=True), sample['label'].cuda(non_blocking=True)

            net_output = model(batch_x)

            batch_accuracy = calculate_accuracy(predicted=net_output, target=batch_y) / 100
            total_accuracy += batch_accuracy
            total_batches += 1

        accuracy = (total_accuracy / total_batches)
        print(f'Accuracy of the network on the {total_batches * args.batch_size} test images: {100 * accuracy} %', flush=True)
        return accuracy


def main():
    # path_to_model = '/nfs/stak/users/morgamat/hpc-share/CS_499/CS_499_Term_Project/ResNet-50-CBAM-PyTorch/pretrained_weights/cifar_10_dataset_clean_models/resnet_cbam/resnet_cbam/20_epoch_model.pt'
    path_to_model = '/nfs/stak/users/morgamat/hpc-share/CS_499/CS_499_Term_Project/ImageNet-Models/pretrained_coatnet.pt'
    assert path_to_model[-3:] == '.pt'

    DATASET_SIZE = 0.02
    IMG_SIZE = 224
    
    model = torch.load(path_to_model)
    model = model.to(device)

    # get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    # train_dataset = LoadDataset(dataset_folder_path=args.data_folder, image_size=IMG_SIZE, image_depth=args.img_depth, train=True,
    #                         transform=transforms.ToTensor())
    test_dataset = LoadDataset(dataset_folder_path=args.data_folder, image_size=IMG_SIZE, image_depth=args.img_depth, train=False,
                                transform=transforms, validate=True)

    # We can train on a subset of the dataset:
    # train_subset_sampler = get_subset_random_sampler(train_dataset, DATASET_SIZE)
    test_subset_sampler = get_subset_random_sampler(test_dataset, DATASET_SIZE)
    
    # train_generator = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
    #                                 num_workers=args.num_workers, pin_memory=True, sampler=train_subset_sampler)
    test_generator = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                    pin_memory=True, sampler=test_subset_sampler)

    validate(model, test_generator)


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    diff_mins = (end_time - start_time) // 60
    print(f'\nTook {diff_mins} total minutes', flush=True)