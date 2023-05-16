# Torch imports
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torchvision.transforms.functional as F

# Util imports
from helper_files import load_dataset as DatasetLoader
from helper_files.runtime_args import args
from helper_files.plot import plot_loss_acc

# Other
import PGD
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'gpu' else 'cpu')
DEBUG = True


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
def generate_adversarial_images(count, model, dataset, niter, epsilon, stepsize, loss, randinit = False):
    cleanImages = []
    advImages = []
    cleanLabels = []
    for i, entry in enumerate(dataset):
        inputs = entry['image'].to(device)
        labels = entry['label'].to(device)

        cleanImages.append(inputs)
        advImages.append(PGD.pgd(inputs, labels, model, niter, epsilon, stepsize, loss, randinit))
        cleanLabels.append(labels)

        if((i + 1) * args.batch_size >= count):
            break

    return torch.cat(cleanImages), torch.cat(advImages), torch.cat(cleanLabels) #Concatenate all adv images into 1 4-d tensor instead of a list of smaller 4-d tensors with 1 block of data in them




# TODO: have better modularity rather than this mess of stuff in the main block
if __name__ == "__main__":
    # Dataset Creation
    dataset = DatasetLoader.LoadDataset(dataset_folder_path=args.data_folder, image_size=args.img_size, image_depth=args.img_depth, train=False,
                            transform=transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                    num_workers=args.num_workers, pin_memory=True)
    
    model = torch.load(args.pretrained_path, map_location=device)
    model.eval()

    

    loss = nn.CrossEntropyLoss().to(device)

    clean, images, labels = generate_adversarial_images(args.PGD_image_count, model, dataloader, args.PGD_niter, args.PGD_epsilon, args.PGD_stepsize, loss)

    if(DEBUG):
            plt.imshow(clean[0].cpu().permute(1,2,0).numpy().squeeze())
            plt.savefig('tests/clean.png')
            plt.imshow(images[0].cpu().permute(1,2,0).numpy().squeeze())
            plt.savefig('tests/adv.png')

    pred_labels = []
    for i in range(0, len(images), args.batch_size):
        thing = images[i:i+args.batch_size]
        _, output = model(images[i:i+args.batch_size])
        predicted = torch.argmax(output, dim=1)
        pred_labels.append(predicted)

    pred_labels = torch.cat(pred_labels)
    correct_pred = torch.sum(pred_labels == labels)
    print(f"Correct: {correct_pred}, Percent: {correct_pred/len(images)}")
    


