#!/bin/bash
# Run this script from the folder "generalization/"

# get directory of this script
file_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $file_dir; cd ../

# python ./utils/train.py --config ./configs/cifar-normal_labels.yaml 

python ./utils/train.py --config ./configs/cifar-alexnet-gaussian_pixels.yaml  
python ./utils/train.py --config ./configs/cifar-alexnet-shuffled_pixels.yaml  
python ./utils/train.py --config ./configs/cifar-alexnet-partial_labels.yaml  

python ./utils/train.py --config ./configs/cifar-inception-gaussian_pixels.yaml  
python ./utils/train.py --config ./configs/cifar-inception-shuffled_pixels.yaml  
python ./utils/train.py --config ./configs/cifar-inception-partial_labels.yaml   

python ./utils/train.py --config ./configs/cifar-resnet-gaussian_pixels.yaml  
python ./utils/train.py --config ./configs/cifar-resnet-shuffled_pixels.yaml
python ./utils/train.py --config ./configs/cifar-resnet-partial_labels.yaml
