# Torch imports
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import os
import torchvision.transforms.functional as F

# Util imports
from helper_files import load_dataset as DatasetLoader
from helper_files.runtime_args import args
from helper_files.plot import plot_loss_acc

# Other
import PGD
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'gpu' else 'cpu')
DEBUG = args.DEBUG

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

# TODO: test this func thoroughly
def generate_adversarial_images(count, model, dataset, niter, epsilon, stepsize, randinit = False):
    loss = nn.CrossEntropyLoss().to(device)
    cleanImages = []
    advImages = []
    cleanLabels = []
    for i, entry in enumerate(dataset):
        inputs = entry['image'].to(device)
        labels = entry['label'].to(device)
               
        cleanImages.append(inputs)
        advImages.append(PGD.pgd(inputs, labels, model, niter, epsilon, stepsize, loss, randinit))
        cleanLabels.append(labels)
        if(DEBUG):
            fig = plt.figure(figsize=(1, 2))
            fig.add_subplot(1, 2, 1)
            plt.imshow(inputs[0].permute(1, 2, 0).cpu().numpy())
            fig.add_subplot(1, 2, 2)
            plt.imshow((advImages[0][0]).permute(1,2,0).cpu().numpy())
            plt.show()
        if((i + 1) * args.batch_size >= count):
            break

    return torch.cat(cleanImages), torch.cat(advImages), torch.cat(cleanLabels) #Concatenate all adv images into 1 4-d tensor instead of a list of smaller 4-d tensors with 1 block of data in them

def save_all_adversarial_images(dirname, images, labels):
    print(labels)
    counts = [0 for _ in range(10)]
    what = range(args.PGD_image_count)
    print(what)
    for idx in tqdm(what):
        label = labels[idx]
        filename = dirname + "/" + str(label.item()) + "/" + str(counts[label.item()]) + ".png"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        save_image(images[idx], filename)
        counts[label] = counts[label] + 1



# TODO: have better modularity rather than this mess of stuff in the main block
if __name__ == "__main__":
    print(torch.cuda.is_available())
    print(device)
    # Dataset Creation
    dataset = DatasetLoader.LoadDataset(dataset_folder_path=args.data_folder, image_size=args.img_size, image_depth=args.img_depth, train=False,
                                transform=transforms.ToTensor())

    sampler = DatasetLoader.get_subset_random_sampler(dataset, 1.0)

    test_generator = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                    pin_memory=True, sampler=sampler)

    #print("Total dataset size:", len(test_generator))
    #num_classes = len(test_generator.classes)
    
    model = torch.load(args.pretrained_path, map_location=device)
    model.eval()
    

    print(f"\n== Generating Adversarial Images with epsilon {args.PGD_epsilon}\n")
    clean, advImages, labels = generate_adversarial_images(args.PGD_image_count, model, test_generator, args.PGD_niter, args.PGD_epsilon, args.PGD_stepsize)
    print("\n== Saving Images to drive\n")
    save_all_adversarial_images(f"{args.PGD_save_path}/epsilon_{args.PGD_epsilon}_niter_{args.PGD_niter}", advImages, labels)
    #save_image(images[0][0], "test.png")
    #print(labels[0])


    pred_labels = []
    for i in range(0, len(advImages), args.batch_size):
        thing = advImages[i:i+args.batch_size]
        _, output = model(advImages[i:i+args.batch_size])
        predicted = torch.argmax(output, dim=1)
        pred_labels.append(predicted)

    pred_labels = torch.cat(pred_labels)
    correct_pred = torch.sum(pred_labels == labels)
    print(f"Correct: {correct_pred}, Percent: {correct_pred/len(advImages)}")
    


