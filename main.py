import argparse
import os
import pdb

import numpy as np
import torch
from torchvision import transforms

import calculate_log as callog
import data_utils
import detectors.baselines as ood_competitors
import detectors.detect_utils
import malcom

detector_dict = {
    'malcom' : malcom.Malcom,
    'baseline' : ood_competitors.Baseline,
    'odin' : ood_competitors.Odin,
    'mahalanobis' : ood_competitors.Mahalanobis}

def main(args):
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu_idx)
    os.environ["CUDA_DEVICE"]=str(args.gpu_idx)

    np.random.seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.set_device(args.gpu_idx)

    pretrained = os.path.join(args.net_dir, '{}_{}.pth'.format(args.net_type, args.dataset))

    # set the out-of-distribution data
    out_dist_list = ['imagenet_crop', 'imagenet_resize', 'lsun_crop', 'lsun_resize', 'isun']
    if args.dataset == 'cifar10':
        out_dist_list = ['cifar100', 'svhn'] + out_dist_list
        input_stds = (0.2470, 0.2435, 0.2616)
        in_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
            input_stds)
        ])
    elif args.dataset == 'cifar100':
        out_dist_list = ['cifar10', 'svhn'] + out_dist_list
        input_stds = (0.2673, 0.2564, 0.2762)
        in_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409),
            input_stds)
        ])
    elif args.dataset == 'svhn':
        out_dist_list = ['cifar10', 'cifar100'] + out_dist_list
        input_stds = (1.0, 1.0, 1.0)
        in_transform = transforms.Compose([transforms.ToTensor()])

    # load model
    print('load model: ' + args.net_type)
    model = torch.load(pretrained, map_location = "cuda:" + str(args.gpu_idx))
    model.cuda()
    model.eval()

    # load dataset
    print('load target data: ' + args.dataset)
    train_loader = data_utils.get_dataloader(
        args.dataset, args.data_root, 'train', in_transform, args.batch_size
    )
    test_loader = data_utils.get_dataloader(
        args.dataset, args.data_root, 'test', in_transform, args.batch_size
    )

    # fit detector
    print('fit detector')
    OOD_Detector = detector_dict[args.detector_type]
    main_detector = OOD_Detector(
        model,
        args.num_classes,
        ood_tuning=args.ood_tuning,
        net_type='' if args.naive_layer else args.net_type,
        normalizer=input_stds if args.detector_type in ['odin', 'mahalanobis']
                              else None,
    )
    if args.detector_type == 'malcom' and args.ood_tuning:
        args.detector_type = 'malcom++'
    main_detector.fit(train_loader)

    # get scores
    print('get scores')
    results = []
    if not args.ood_tuning:
        in_scores = detectors.detect_utils.get_scores(main_detector, test_loader)
    for _, out_dist in enumerate(out_dist_list):
        print('\t...out-of-distribution: ' + out_dist)
        out_test_loader = data_utils.get_dataloader(
            out_dist, args.data_root, 'test', in_transform, args.batch_size
        )
        
        if args.ood_tuning:
            main_detector.tune_parameters(
                test_loader, out_test_loader, num_samples=1000
            )
            in_scores = detectors.detect_utils.get_scores(
                main_detector, test_loader
            )
            out_scores = detectors.detect_utils.get_scores(
                main_detector, out_test_loader
            )
        else:
            out_scores = detectors.detect_utils.get_scores(
                main_detector, out_test_loader
            )
        test_results = callog.metric(-in_scores[1000:], -out_scores[1000:])
        results.append(test_results)

    mtypes = ['', 'TNR', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']
    print('='*78)
    print('{} detector (with {} trained on {} '.format(
        args.detector_type, args.net_type, args.dataset
        ), end='')
    print('w/o using ood samples): ' if not args.ood_tuning else 'with using ood samples): ')
    for mtype in mtypes:
        print(' {mtype:^12s}'.format(mtype=mtype), end='')
    for count_out, result in enumerate(results) :
        print('\n {:12}'.format(out_dist_list[count_out][:10]), end='')
        print(' {val:^12.2f}'.format(val=100.*result['TNR']), end='')
        print(' {val:^12.2f}'.format(val=100.*result['AUROC']), end='')
        print(' {val:^12.2f}'.format(val=100.*result['DTACC']), end='')
        print(' {val:^12.2f}'.format(val=100.*result['AUIN']), end='')
        print(' {val:^12.2f}'.format(val=100.*result['AUOUT']), end='')
    print('')
    print('='*78)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train CNNs for ID classification')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for data loader')
    parser.add_argument('--data_root', default='./data', help='path to dataset')
    parser.add_argument('--dataset', required=True, help='cifar10 | cifar100 | svhn')
    parser.add_argument('--detector_type', type=str, default='malcom', help='OOD detector')
    parser.add_argument('--gpu_idx', type=int, default=0, help='gpu index')
    parser.add_argument('--naive_layer', type=int, default=0)
    parser.add_argument('--net_type', required=True, help='resnet | densenet | vanilla')
    parser.add_argument('--num_classes', type=int, default=10, help='the # of classes')
    parser.add_argument('--net_dir', default='./pretrained', help='path for trained model')
    parser.add_argument('--ood_tuning', type=int, default=0, help='use OOD samples to tune parameters')


    args = parser.parse_args()
    if not args.net_type in ['densenet', 'resnet', 'vanilla']:
        raise ValueError('Invalid model type')

    if args.dataset == 'cifar100':
        args.num_classes = 100
    elif not args.dataset in ['cifar10', 'svhn']:
        raise ValueError('Invalid dataset')

    if not args.detector_type in (list(detector_dict.keys()) + ['malcom']):
        raise ValueError('Invalid detector')

    if args.detector_type in ['odin', 'mahalanobis'] and args.ood_tuning == 0:
        raise ValueError('Invalid setting w/o using OODs. ' + \
             'Set the argument "ood_tuning" as 1.')

    if not os.path.exists(args.net_dir):
        raise ValueError('Invalid pretrained network directory')
    print(args)
    
    main(args)

