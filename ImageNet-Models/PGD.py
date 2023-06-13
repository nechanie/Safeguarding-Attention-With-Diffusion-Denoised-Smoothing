import json
import time
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import os
import torchvision.transforms.functional as F

import load_dataset as DatasetLoader
from runtime_args import args

import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'gpu' else 'cpu')
DEBUG = args.DEBUG


def pgd(input, labels, model, iters, epsilon, stepsize, loss = None):
    input_copy = input.detach().clone() # Clones to create new set of adversarial images

    if(loss == None):
        loss = nn.CrossEntropyLoss()

    for _ in range(iters):
        input_copy.requires_grad = True
        model.zero_grad()
        pred = model(input_copy)
        loss_obj = loss(pred, labels)
        loss_obj.backward()                      
        grad = input_copy.grad.detach()
        grad = grad.sign()
        input_copy = input_copy + stepsize * grad

        # Project x_copy onto x to get our adversarial x
        input_copy = input + torch.clamp(input_copy - input, min=-epsilon, max=epsilon)
        input_copy = input_copy.detach()
        input_copy = torch.clamp(input_copy, min=0, max=1)

    return input_copy



# TODO: test this func thoroughly
def generate_adversarial_images(count, model, dataset, niter, epsilon, stepsize, save_path, randinit = False):
    counts = [0 for _ in range(1000)]
    class_map = json.load(open('imagenet_class_index.json', 'r'))
    loss = nn.CrossEntropyLoss().to(device)
    cleanImages = []
    advImages = []
    cleanLabels = []
    for i, entry in tqdm(enumerate(dataset)):
        inputs = entry['image'].to(device)
        labels = entry['label'].to(device)
               
        # cleanImages.append(inputs)
        # advImages.append(pgd(inputs, labels, model, niter, epsilon, stepsize, loss))
        # cleanLabels.append(labels)

        save_all_adversarial_images(save_path, pgd(inputs, labels, model, niter, epsilon, stepsize, loss), labels, class_map, counts)

    # return torch.cat(cleanImages), torch.cat(advImages), torch.cat(cleanLabels) #Concatenate all adv images into 1 4-d tensor instead of a list of smaller 4-d tensors with 1 block of data in them




def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    fig.show()


def save_all_adversarial_images(dirname, images, labels, class_map, counts):
    
    for idx in range(len(images)):
        label = labels[idx]
        new_label = class_map[str(label.item())][0]
        filename = dirname + "/" + new_label + "/" + str(counts[label.item()]) + ".png"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        save_image(images[idx], filename)
        counts[label] = counts[label] + 1
