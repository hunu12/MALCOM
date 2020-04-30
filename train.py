
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from data_utils import get_dataloader
import models

def evaluation(model, test_loader, criterion):
    model.eval()
    total_loss, total_correct, total = 0.0, 0.0, 0
    for data, labels in test_loader:
        data = data.cuda()
        labels = labels.cuda()
        total += data.size(0)
        with torch.no_grad():
            outputs = model(data)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            total_loss += loss.item() * data.size(0)
            total_correct += torch.sum(preds == labels.data)
    return total_loss / total, total_correct / total

def main(args):
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu_idx)
    os.environ["CUDA_DEVICE"]=str(args.gpu_idx)

    np.random.seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.set_device(args.gpu_idx)

    out_file = os.path.join(args.output_dir, '{}_{}.pth'.format(args.net_type, args.dataset))
    
    # set the transformations for training
    tfs_for_augmentation = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),]
    if args.dataset == 'cifar10':
        train_transform = transforms.Compose(
            tfs_for_augmentation +
            [transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4821, 0.4465), (0.2470, 0.2435, 0.2616)),])
    elif args.dataset == 'cifar100':
        train_transform = transforms.Compose(
            tfs_for_augmentation +
            [transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),])
    elif args.dataset == 'svhn':
        train_transform = transforms.Compose([transforms.ToTensor()])
        test_transform = transforms.Compose([transforms.ToTensor()])

    # load model
    if args.net_type == 'densenet':
        if args.dataset == 'svhn':
            model = models.DenseNet3(100, args.num_classes, growth_rate=12, dropRate=0.2)    
        else:
            model = models.DenseNet3(100, args.num_classes, growth_rate=12)    
    elif args.net_type == 'resnet':
        model = models.ResNet34(num_c=args.num_classes)
    elif args.net_type == 'vanilla':
        model = models.VanillaCNN(args.num_classes)
    model.cuda()
    print('load model: ' + args.net_type)

    # load dataset
    print('load target data: ' + args.dataset)
    if args.dataset == 'svhn':
        train_loader, valid_loader = get_dataloader(
            args.dataset, args.data_root, 'train', train_transform, args.batch_size, valid_transform=test_transform)
    else:
        train_loader = get_dataloader(args.dataset, args.data_root, 'train', train_transform, args.batch_size)
    test_loader = get_dataloader(args.dataset, args.data_root, 'test', test_transform, args.batch_size)

    # define objective and optimizer
    criterion = nn.CrossEntropyLoss()
    if args.net_type == 'densenet' or args.net_type == 'vanilla':
        weight_decay = 1e-4
        milestones = [150, 225]
        gamma = 0.1
    elif args.net_type == 'resnet':
        weight_decay = 5e-4
        milestones = [60, 120, 160]
        gamma = 0.2
    if args.dataset == 'svhn' or args.net_type == 'vanilla':
        milestones = [20, 30]

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=gamma)

    # train
    best_loss = np.inf
    iter_cnt = 0
    for epoch in range(args.num_epochs):
        model.train()
        total, total_loss, total_step = 0, 0, 0
        for itr, (data, labels) in enumerate(train_loader):
            data = data.cuda()
            labels = labels.cuda()
            total += data.size(0)
            
            outputs = model(data)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * data.size(0)
            iter_cnt += 1
            total_step += 1

            if args.dataset=='svhn' and iter_cnt >= 200:
                valid_loss, _ = evaluation(model, valid_loader, criterion)
                test_loss, acc = evaluation(model, test_loader, criterion)
                print('Epoch [{:03d}/{:03d}], step [{}/{}] train loss : {:.4f}, valid loss : {:.4f}, test loss : {:.4f}, test acc : {:.2f} %'
                  .format(epoch+1, args.num_epochs, total_step, len(train_loader), total_loss / total, valid_loss, test_loss, 100 * acc))
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    torch.save(model, out_file)
                iter_cnt = 0
                model.train()

        if args.dataset != 'svhn':
            test_loss, acc = evaluation(model, test_loader, criterion)
            print('[{:03d}/{:03d}] train loss : {:.4f}, test loss : {:.4f}, test acc : {:.2f} %'
              .format(epoch+1, args.num_epochs, total_loss / total, test_loss, 100 * acc))
            torch.save(model, out_file)

        scheduler.step()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train CNNs for ID classification')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for data loader')
    parser.add_argument('--data_root', default='./data', help='path to dataset')
    parser.add_argument('--dataset', required=True, help='cifar10 | cifar100 | svhn')
    parser.add_argument('--gpu_idx', type=int, default=0, help='gpu index')
    parser.add_argument('--net_type', required=True, help='resnet | densenet | vanilla')
    parser.add_argument('--num_classes', type=int, default=10, help='the # of classes')
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--output_dir', default='./pretrained', help='output path for trained model')

    args = parser.parse_args()
    if args.net_type == 'densenet':
        args.batch_size = 64
        args.num_epochs = 300
    elif args.net_type == 'resnet':
        args.batch_size = 128
        args.num_epochs = 200
    elif args.net_type != 'vanilla':
        raise ValueError('Invalid model type')

    if args.dataset == 'svhn':
        args.num_epochs = 40
    elif args.dataset == 'cifar100':
        args.num_classes = 100
    elif args.dataset != 'cifar10':
        raise ValueError('Invalid dataset')
    print(args)

    start_time = time.time()
    main(args)
    print("--- training time : {} seconds ---".format(time.time() - start_time))
