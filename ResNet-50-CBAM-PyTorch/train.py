'''Training script.
'''

from datetime import datetime
import os
import time
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
from torchsummary import summary
from torchvision import transforms
import torch.distributed as dist
import torch.multiprocessing as mp

from models.resnet50 import ResNet50
from Util.runtime_args import args
from Util.load_dataset import LoadDataset, get_subset_random_sampler
from Util.plot import plot_loss_acc
from Util.helpers import calculate_accuracy


now = datetime.now()
FORMATTED_DATETIME = now.strftime("%Y-%-m-%-d__%-H-%-M-%-S")
DIR_PATH = f'runs/{FORMATTED_DATETIME}/'

os.mkdir(DIR_PATH)

DATASET_SIZE = 1.0

# device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'gpu' else 'cpu')

graphs_folder = DIR_PATH + args.graphs_folder
if not os.path.exists(graphs_folder): 
    os.mkdir(graphs_folder)

model_save_folder = DIR_PATH + 'resnet_cbam/' if args.use_cbam else DIR_PATH + 'resnet/'
if not os.path.exists(model_save_folder):
    os.mkdir(model_save_folder)


def print_timing_metrics(start_time, epoch_idx):
    end_time = time.time()
    time_diff = end_time - start_time
    time_per_epoch_mins = (time_diff / (epoch_idx + 1)) / 60
    time_remaining_mins = (args.epoch - (epoch_idx + 1)) * time_per_epoch_mins
    print(f"{time_remaining_mins:.2f} mins remaining... {time_per_epoch_mins:.2f} per epoch avg.", flush=True)


def train(gpu, args):
    '''Init models and dataloaders and train/validate model.
    '''

    # rank = args.rank * args.gpus + gpu
    # world_size = args.gpus * args.nodes


    # dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)

    model = ResNet50(image_depth=args.img_depth, num_classes=args.num_classes, use_cbam=args.use_cbam)
    torch.cuda.set_device(gpu)
    model.cuda(gpu)

    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    lr_decay = lr_scheduler.ExponentialLR(optimizer, gamma=args.decay_rate)
    criterion = torch.nn.CrossEntropyLoss().cuda(gpu)

    print(f"Model details:\n" \
        + f"Opt: {optimizer}\n" \
        + f"Lr_decay: {lr_decay}\n" \
        + f"Crit: {criterion}\n" \
        + f"Batch_size: {args.batch_size}\n" \
        + f"Epochs: {args.epoch}\n" \
        + f"Dataset: {args.data_folder}\n" \
        + f"Dataset Size: {DATASET_SIZE}\n" \
        + f"Cbam?: {args.use_cbam}\n",
        flush=True
    )

    summary(model, (3, 224, 224))


    # model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    train_dataset = LoadDataset(dataset_folder_path=args.data_folder, image_size=args.img_size, image_depth=args.img_depth, train=True,
                            transform=transforms.ToTensor())
    test_dataset = LoadDataset(dataset_folder_path=args.data_folder, image_size=args.img_size, image_depth=args.img_depth, train=False,
                                transform=transforms.ToTensor())

    # We can train on a subset of the dataset:
    train_subset_sampler = get_subset_random_sampler(train_dataset, DATASET_SIZE)
    test_subset_sampler = get_subset_random_sampler(test_dataset, DATASET_SIZE)

    # train_distributed_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    # test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)

    # train_composite_sampler = torch.utils.data.sampler.CompositeSampler([train_subset_sampler, train_distributed_sampler])
    # test_composite_sampler = torch.utils.data.sampler.CompositeSampler([test_subset_sampler, test_distributed_sampler])

    train_generator = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers, pin_memory=True, sampler=train_subset_sampler)
    test_generator = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                    pin_memory=True, sampler=test_subset_sampler)

    start_time = time.time()

    training_loss_list = []
    training_acc_list = []
    testing_loss_list = []
    testing_acc_list = []

    best_accuracy = 0
    for epoch_idx in range(args.epoch):
        print(f"Epoch {epoch_idx}", flush=True)

        #Model Training & Validation.
        model.train()

        epoch_loss = []
        epoch_accuracy = []
        i = 0

        for i, sample in tqdm(enumerate(train_generator), total=len(train_generator)):

            batch_x, batch_y = sample['image'].cuda(non_blocking=True), sample['label'].cuda(non_blocking=True)

            optimizer.zero_grad()

            _,net_output = model(batch_x)
            total_loss = criterion(input=net_output, target=batch_y)

            total_loss.backward()
            optimizer.step()
            batch_accuracy = calculate_accuracy(predicted=net_output, target=batch_y)
            epoch_loss.append(total_loss.item())
            epoch_accuracy.append(batch_accuracy)

        curr_accuracy = sum(epoch_accuracy)/(i+1)
        curr_loss = sum(epoch_loss)/(i+1)

        training_loss_list.append(curr_loss)
        training_acc_list.append(curr_accuracy)

        print(f"Training Loss : {curr_loss}, Training accuracy : {curr_accuracy}", flush=True)

        model.eval()
        epoch_loss = []
        epoch_accuracy = []
        i = 0

        with torch.set_grad_enabled(False):
            for i, sample in tqdm(enumerate(test_generator), total=len(test_generator)):

                batch_x, batch_y = sample['image'].cuda(non_blocking=True), sample['label'].cuda(non_blocking=True)

                _,net_output = model(batch_x)

                total_loss = criterion(input=net_output, target=batch_y)

                batch_accuracy = calculate_accuracy(predicted=net_output, target=batch_y)
                epoch_loss.append(total_loss.item())
                epoch_accuracy.append(batch_accuracy)

            curr_accuracy = sum(epoch_accuracy)/(i+1)
            curr_loss = sum(epoch_loss)/(i+1)

            testing_loss_list.append(curr_loss)
            testing_acc_list.append(curr_accuracy)

        print(f"Testing Loss : {curr_loss}, Testing accuracy : {curr_accuracy}", flush=True)

        #plot accuracy and loss graph
        plot_loss_acc(path=graphs_folder, num_epoch=epoch_idx, train_accuracies=training_acc_list, train_losses=training_loss_list,
                            test_accuracies=testing_acc_list, test_losses=testing_loss_list)

        if epoch_idx % 5 == 0:

            lr_decay.step() #decrease the learning rate at every n epoch.
            curr_lr = 0
            for params in optimizer.param_groups:
                curr_lr = params['lr']
            print(f"The current learning rate for training is : {curr_lr}", flush=True)


        if (epoch_idx + 1) % 5 == 0 or epoch_idx == args.epoch - 1:
            torch.save(model, f"{model_save_folder}{epoch_idx + 1}_epoch_model.pt")
            best_accuracy = curr_accuracy
            print('Model is saved!', flush=True)

        print_timing_metrics(start_time, epoch_idx)

        print('\n--------------------------------------------------------------------------------\n', flush=True)


os.environ['MASTER_ADDR'] = '10.192.44.201'
os.environ['MASTER_PORT'] = '8374'

if __name__ == '__main__':
    # I couldn't get the distributed training to work, so I commented it out
    # I imagine it is due to the ip address of the server listed above ^
    # mp.spawn(train, nprocs=args.gpus, args=(args,))
    train(0, args)

