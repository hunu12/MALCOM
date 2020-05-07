""" original code is from
[1] https://github.com/pokaxpoka/deep_Mahalanobis_detector/blob/master/data_loader.py
[2] https://github.com/aaron-xichen/pytorch-playground
"""
import os

import numpy as np
import torch
from torchvision import datasets
from torch.utils.data import DataLoader, TensorDataset

def subsample_dataset(dataset, num_samples):
    num_classes = len(np.unique(dataset.labels))

    target_indices = np.zeros(len(dataset), dtype=bool)
    for c in range(num_classes):
        class_indices = np.where(dataset.labels==c)[0]
        choice = np.random.choice(
            class_indices, size=num_samples, replace=False
        )
        target_indices[choice] = True
    return target_indices

def getSVHN(data_root, split, transforms, batch_size, 
            valid_transform=None, **kwargs):
    assert split in ['train', 'test']
    data_root = os.path.expanduser(os.path.join(data_root, 'svhn-data'))
    dataset = datasets.SVHN(
                root=data_root, split=split, download=True,
                transform=transforms,
    )
    if valid_transform is None:
        data_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=(split=='train'), **kwargs
        )
        return data_loader
    
    extra_dataset = datasets.SVHN(
            root=data_root, split='extra', download=True,
            transform=transforms,
    )

    valid_from_train = subsample_dataset(dataset, 400)
    valid_from_extra = subsample_dataset(extra_dataset, 200)

    train_X = torch.from_numpy(dataset.data[~valid_from_train]).float() / 255.0
    train_Y = torch.from_numpy(dataset.labels[~valid_from_train])

    valid_X = torch.from_numpy(
        np.concatenate((
            dataset.data[valid_from_train], 
            extra_dataset.data[valid_from_extra]), axis=0)
    ).float() / 255.0
    valid_Y = torch.from_numpy(
        np.concatenate((
            dataset.labels[valid_from_train],
            extra_dataset.labels[valid_from_extra]), axis=0)
    )

    train_dataset = TensorDataset(train_X, train_Y)
    valid_dataset = TensorDataset(valid_X, valid_Y)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    return train_loader, valid_loader

def get_dataloader(data_type, data_root, split, transforms, batch_size,
                   valid_transform=None, **kwargs):
    assert split in ['train', 'test']

    if data_type == 'cifar10':
        data_root = os.path.expanduser(os.path.join(data_root, 'cifar10-data'))
        dataset = datasets.CIFAR10(
            root=data_root, train=(split=='train'),
            download=True, transform=transforms
        )
    elif data_type == 'cifar100':
        data_root = os.path.expanduser(os.path.join(data_root, 'cifar100-data'))
        dataset = datasets.CIFAR100(
            root=data_root, train=(split=='train'),
            download=True, transform=transforms
        )
    elif data_type == 'svhn':
        data_loader = getSVHN(data_root, split, transforms, batch_size, valid_transform=valid_transform, **kwargs)
        return data_loader
    elif data_type == 'imagenet_crop':
        data_root = os.path.expanduser(os.path.join(data_root, 'Imagenet'))
        dataset = datasets.ImageFolder(data_root, transform=transforms)
    elif data_type == 'imagenet_resize':
        data_root = os.path.expanduser(os.path.join(data_root, 'Imagenet_resize'))
        dataset = datasets.ImageFolder(data_root, transform=transforms)
    elif data_type == 'lsun_crop':
        data_root = os.path.expanduser(os.path.join(data_root, 'LSUN'))
        dataset = datasets.ImageFolder(data_root, transform=transforms)
    elif data_type == 'lsun_resize':
        data_root = os.path.expanduser(os.path.join(data_root, 'LSUN_resize'))
        dataset = datasets.ImageFolder(data_root, transform=transforms)
    elif data_type == 'isun':
        data_root = os.path.expanduser(os.path.join(data_root, 'iSUN'))
        dataset = datasets.ImageFolder(data_root, transform=transforms)
    else:
        raise ValueError("Invalid dataset type")
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=(split=='train')
    )
    return data_loader