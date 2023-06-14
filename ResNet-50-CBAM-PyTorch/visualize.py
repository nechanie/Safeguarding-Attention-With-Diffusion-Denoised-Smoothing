'''
Visualize the trained model's feature maps.
'''

import os
from tqdm import tqdm
from collections import OrderedDict
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

from load_dataset import LoadDataset, get_subset_random_sampler
from models.resnet50 import ResNet50

from runtime_args import args

print("Use cbam:", args.use_cbam)

device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'gpu' else 'cpu')

assert args.pretrained_path and os.path.exists(args.pretrained_path), 'A trained model does not exist!'
model = torch.load(args.pretrained_path)
model = model.to(device)
print("Model loaded!")


model.eval()

input_dataset = LoadDataset(dataset_folder_path=args.data_folder, image_size=args.img_size, image_depth=args.img_depth, 
                            validate=False, train=False, transform=transforms.ToTensor())

print("Dataset Size:", len(input_dataset))
subset_sampler = get_subset_random_sampler(input_dataset, dataset_len=50)

data_generator = DataLoader(input_dataset, batch_size=1, shuffle=False, num_workers=1, sampler=subset_sampler)

print("Subset Size:", len(data_generator))

# class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

pretrained_dir = '/'.join(args.pretrained_path.split('/')[:-1])
output_folder = f'./vis_output_resnet_cbam' if args.use_cbam else f'./vis_output_resnet'

if not os.path.exists(output_folder) : os.mkdir(output_folder)


fig = plt.figure(figsize=(10, 4))

for i, sample in tqdm(enumerate(data_generator), total=len(data_generator)):
    image, true_label_name = sample['image'], class_names[sample['label'].item()]

    plt.clf()

    image = image.to(device)

    cnn_filters, output = model(image)

    #identify the predicted class
    softmaxed_output = torch.nn.Softmax(dim=1)(output)
    predicted_class = class_names[torch.argmax(softmaxed_output).cpu().numpy()]


    #merge all the filters together as one and resize them to the original image size for viewing.
    # attention_combined_filter = cv2.resize(torch.max(attention_filters.squeeze(0), 0)[0].detach().numpy(), (args.img_size, args.img_size))
    cnn_combined_filter = cv2.resize(torch.max(cnn_filters.squeeze(0), 0)[0].detach().cpu().numpy(), (args.img_size, args.img_size))
    heatmap = np.asarray(cv2.applyColorMap(cv2.normalize(cnn_combined_filter, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U),
                        cv2.COLORMAP_JET), dtype=np.float32)


    input_img = cv2.resize(image.squeeze(0).permute(1, 2, 0).cpu().numpy(), (args.img_size, args.img_size))

    #create heatmap by overlaying the filters on the original image
    heatmap_cnn = cv2.addWeighted(np.asarray(input_img, dtype=np.float32), 0.9, heatmap, 0.0025, 0)

    fig.add_subplot(131)
    plt.imshow(input_img)
    plt.title("Input Image")
    plt.xticks(())
    plt.yticks(())

    fig.add_subplot(132)
    plt.imshow(cnn_combined_filter)
    if args.use_cbam:
        plt.title("CNN Feature Map with CBAM")
    else:
        plt.title("CNN Feature Map without CBAM")

    plt.xticks(())
    plt.yticks(())

    fig.add_subplot(133)
    plt.imshow(heatmap_cnn)
    plt.title("Heat Map")
    plt.xticks(())
    plt.yticks(())

    fig.suptitle(f"Network's prediction : {predicted_class.capitalize()}\nTrue label: {true_label_name.capitalize()}", fontsize=16)

    plt.savefig(f'{output_folder}/{i}.png')
