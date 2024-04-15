# Understanding Deep Learning Requires Rethinking Generalization

>Reproducing results from the 2017 paper [1611.03530](https://arxiv.org/abs/1611.03530)

## This Repo contributes:
- A modular approach to using datasets with randomization tests.
- Implements all models described in the paper, both in Flax and Pytorch.
- Reproduces original experiments. See this [report](https://wandb.ai/stepp1/generalization-2/reports/Understanding-Rethinking-Generalization-CIFAR10--Vmlldzo0NTA3Njcz).


## Contributions

* **Randomization tests**: *Deep neural networks easily fit random labels.*
>  At the heart of our methodology is a variant of the well-known randomization test from non-parametric statistics (Edgington & Onghena, 2007). In a first set of experiments, we train several standard architectures on a copy of the data where the true labels were replaced by random labels.

* **The role of explicit regularization**: *Explicit regularization may improve generalization performance, but is neither necessary nor by itself sufficient for controlling generalization error.*
> If the model architecture itself isn’t a sufficient regularizer, it remains to see how much explicit regularization helps. We show that explicit forms of regularization, such as weight decay, dropout, and data augmentation, do not adequately explain the generalization error of neural networks.

* **Finite sample expressivity**. 
> We complement our empirical observations with a theoretical construction showing that **generically large neural networks can express any labeling of the training data**. More formally, we exhibit a very simple two-layer ReLU network with p = 2n + d parameters that can express any labeling of any sample of size n in d dimensions. A previous construction due to Livni et al. (2014) achieved a similar result with far more parameters, namely, O(dn). While our depth 2 network inevitably has large width, we can also come up with a depth k network in which each layer has only O(n/k) parameters

* **The role of implicit regularization**. 
> While explicit regularizers like dropout and weight-decay may not be essential for generalization, it is certainly the case that not all models that fit the training data well generalize well. Indeed, in neural networks, we almost always choose our model as the output of running stochastic gradient descent. Appealing to linear models, we analyze how SGD acts as an implicit regularizer. For linear models, SGD always converges to a solution with small norm. Hence, the algorithm itself is implicitly regularizing the solution. Indeed, we show on small data sets that even Gaussian kernel methods can generalize well with no regularization. Though this doesn’t explain why certain architectures generalize better than other architectures, it does suggest that more investigation is needed to understand exactly what the properties are inherited by models that were trained using SGD.

## Experiments: Fitting Random Labels and Pixels
We run our experiments with the following modifications of the labels and input images:

| Experiment Name | Description |
| --------------- | ----------- |
| **True Labels**                | Original dataset without modification |
| **Partially Corrupted Labels** | Independently with probability p, the label of each image is corrupted as a uniform random class |
| **Random Labels**              | All the labels are replaced with random ones |
| **Shuffled Pixels**            | A random permutation of the pixels is chosen and then the same permutation is applied to all the images in both training and test set |
| **Random Pixels**              | A different random permutation is applied to each image independently |
| **Gaussian**                   | A Gaussian distribution (with matching mean and variance to the original image dataset) is used to generate random pixels for each image |


## Installation

First, install [torch](https://pytorch.org/get-started/locally/) and [torchvision](https://pytorch.org/get-started/locally/). Then, install the package using pip:


### From PyPI

To install the latest release from PyPI:

```bash
pip install -U ids-generalization
```

### Latest version from GitHub

To pull and install the latest commit from this repository, along with its Python dependencies:

```bash
pip install git+https://github.com/ids-uchile/Rethinking-Generalization.git
```