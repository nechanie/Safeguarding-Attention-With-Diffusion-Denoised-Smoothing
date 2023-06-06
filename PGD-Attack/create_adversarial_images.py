# Torch imports
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# Util imports
from helper_files import load_dataset as DatasetLoader
from helper_files.runtime_args import args

# Other
import PGD
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'gpu' else 'cpu')
DEBUG = True

# TODO: test this func thoroughly
def generate_adversarial_images(count, model, dataset, niter, epsilon, stepsize, randinit = False):
    loss = nn.CrossEntropyLoss().to(device)
    images = []
    outLabels = []
    for i, entry in enumerate(dataset):
        inputs = entry['image'].to(device)
        labels = entry['label'].to(device)
               
        images.append(PGD.pgd(inputs, labels, model, niter, epsilon, stepsize, loss, randinit))
        outLabels.append(labels)
        if(DEBUG):
            fig = plt.figure(figsize=(1, 2))
            fig.add_subplot(1, 2, 1)
            plt.imshow(inputs[0].permute(1, 2, 0).cpu().numpy())
            fig.add_subplot(1, 2, 2)
            plt.imshow((images[0][0]).permute(1,2,0).cpu().numpy())
            plt.show()
        if((i + 1) * args.batch_size >= count):
            break

    return (images, outLabels)




# TODO: have better modularity rather than this mess of stuff in the main block
if __name__ == "__main__":
    print(torch.cuda.is_available())
    print(device)
    # Dataset Creation
    dataset = DatasetLoader.LoadDataset(dataset_folder_path=args.data_folder, image_size=args.img_size, image_depth=args.img_depth, train=False,
                            transform=transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                    num_workers=args.num_workers, pin_memory=True)
    
    model = torch.load(args.pretrained_path, map_location=device)
    model.eval()
    


    images, labels = generate_adversarial_images(args.PGD_image_count, model, dataloader, args.PGD_niter, args.PGD_epsilon, args.PGD_stepsize)
    #save_image(images[0][0], "test.png")
    #print(labels[0])
    

