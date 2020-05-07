# MALCOM
**MA**ha**L**anobis distance with **COM**pression complexity pooling (**MALCOM**) is an out-of-distribution image detector with already deployed convolutional neural networks. This is the project for the paper "Convolutional Neural Networks with Compression Complexity Pooling for Out-of-Distribution Image Detection". The codes are mainly from **[deep-Mahalanobis-detector](https://github.com/pokaxpoka/deep_Mahalanobis_detector)**.

## Getting Started
This project was conducted on Ubuntu 16.04 LTS with
- Nvidia GPU (TITAN Xp) + **CUDA Toolkit 10.2**
- Python 3.6
- Anaconda3

### Prerequisites
- **[PyTorch](http://pytorch.org/)** >= 1.5
- **[PyCUDA](https://mathema.tician.de/software/pycuda/)** >= 2019.1.2
- [scikit-learn](https://mathema.tician.de/software/pycuda/) >= 0.22.1
- [scipy](https://www.scipy.org/) >= 1.4.1

We highly recommend to use Anaconda3 for installation of packages. An example of terminal commands for installation is as follows:
```
$ conda create -n malcom python=3.6
$ conda activate malcom
$ conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
$ conda install scikit-learn scipy
$ python -m pip install pycuda
```
Note that the instruction for the installation of PyTorch depends on the version of Nvidia CUDA toolkit on the local machine. Please follow the installation guide from **[here](https://pytorch.org/get-started/locally/)**.

### Downloading Out-of-Distribution Datasets
We use download links of out-of-distribution datasets from **[ODIN](https://github.com/facebookresearch/odin)**:

- **[Tiny-ImageNet (crop)](https://www.dropbox.com/s/avgm2u562itwpkl/Imagenet.tar.gz)**
- **[Tiny-ImageNet (resize)](https://www.dropbox.com/s/kp3my3412u5k9rl/Imagenet_resize.tar.gz)**
- **[LSUN (crop)](https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz)**
- **[LSUN (resize)](https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz)**
- **[iSUN](https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz)**

Please place them to `./data/`.

### Downloading Image Classifiers
We provide 9 pre-trained convolutional neural networks: VanillaCNN, DenseNet, and ResNet, where models are trained on CIFAR-10, CIFAR-100, and SVHN. Here, VanillaCNN is a simple convolutional neural network which consists of 4 convolutional layers.
- **[VanillaCNN on CIFAR-10](https://drive.google.com/open?id=1-q54dBXW6_Shd3LG-8AVC42PSbqlCBcb)** / **[VanillaCNN on CIFAR-100](https://drive.google.com/open?id=1WwQjFDYGod9tFfmNkAW9DBG2kI-B9jiF)** / **[VanillaCNN on SVHN](https://drive.google.com/open?id=1uB76bGo3hmnlNSTKKmsaXy5LPYcff09o)**
- **[DenseNet on CIFAR-10](https://drive.google.com/open?id=10lC9Ro_fWyuva4W5PwyMWYsxBMbjNHRv)** / **[DenseNet on CIFAR-100](https://drive.google.com/open?id=1eMy3TvybX6mUi4de71aOp8n84LZvRsui)** / **[DenseNet on SVHN](https://drive.google.com/open?id=1x-qAgZuA0biYqrYMEnuvH8IRs5p7aQek)**
- **[ResNet on CIFAR-10](https://drive.google.com/open?id=1UQSIaPm63w8_nf43CUqsFGpppbnCohAx)** / **[ResNet on CIFAR-100](https://drive.google.com/open?id=1Lu1xEAMgCsk89xsODKYTzQv0jOwNWoyk)** / **[ResNet on SVHN](https://drive.google.com/open?id=1tB_P-iBWVStGDSig4ca07OPjG4rIxWs_)**

Please place them to `./pretrained/`.

### * Training Classifiers from Scratch
We provide the training script `train.py` for the image classifiers. Please consider two mandatory input arguments:
* `args.net_type` : `vanilla` \| `densenet` \| `resnet`
* `args.dataset` : `cifar10` \| `cifar100` \| `svhn`

An example of training VanillaCNN for classifying CIFAR-10 is:
```
$ python train.py --net_type vanilla --dataset cifar10
```
The training time in this case(vanilla-cifar10) is less than 30 minutes. For repeated experiments, please change the seed in the script to train different classifiers.


## Usage

### Out-of-Distribution Image Detection with MALCOM
- **MALCOM** (without using OOD test samples)
```
$ python main.py --net_type vanilla --dataset cifar10
```
An example of output is:
```
==============================================================================
malcom detector (with vanilla trained on cifar10 w/o using ood samples): 
                  TNR         AUROC        DTACC         AUIN        AUOUT    
 cifar100        16.83        64.60        60.67        61.50        65.29    
 svhn            74.71        95.27        90.05        92.14        96.86    
 imagenet_c      99.97        99.76        98.63        99.82        99.65    
 imagenet_r      93.74        98.66        94.40        98.64        98.68    
 lsun_crop       99.80        99.59        97.92        99.68        99.38    
 lsun_resiz      95.77        99.02        95.43        99.09        98.91    
 isun            93.16        98.65        94.23        98.87        98.41    
==============================================================================
```

- **MALCOM++** (with using OOD test samples)
```
$ python main.py --net_type vanilla --dataset cifar10 --ood_tuning 1
```

### Out-of-Distribution Image Detection with Other Competitors


- [**Baseline**(ICLR'17, Hendrycks & Gimpel)](https://arxiv.org/abs/1610.02136)
```
$ python main.py --net_type vanilla --dataset cifar10 --detector_type baseline
```

- [**ODIN**(ICLR'18, Liang et al.)](https://arxiv.org/abs/1706.02690)
```
$ python main.py --net_type vanilla --dataset cifar10 --detector_type odin --ood_tuning 1
```

- [**Mahalanobis**(NeurIPS'18, Lee et al.)](https://arxiv.org/abs/1807.03888)
```
$ python main.py --net_type vanilla --dataset cifar10 --detector_type mahalanobis --ood_tuning 1
```

Note that the argument `ood_tuning` should be set as 1 for ODIN and Mahalanobis detector.