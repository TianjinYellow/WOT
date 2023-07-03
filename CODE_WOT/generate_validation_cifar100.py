import torch
import torchvision
import numpy as np

np.random.seed(0)
m = 50000
P = np.random.permutation(m)

n = 2000


def cifar100(root):
    train_set = torchvision.datasets.CIFAR100(root=root, train=True, download=True)
    test_set = torchvision.datasets.CIFAR100(root=root, train=False, download=True)
    return {
        'train': {'data': train_set.data, 'labels': train_set.targets},
        'test': {'data': test_set.data, 'labels': test_set.targets}
    }


dataset = cifar100('../cifar100-data')

val_data = dataset['train']['data'][P[:n]]
val_labels = [dataset['train']['labels'][p] for p in P[:n]]
train_data = dataset['train']['data'][P[n:]]
train_labels = [dataset['train']['labels'][p] for p in P[n:]]

dataset['train']['data'] = train_data
dataset['train']['labels'] = train_labels
dataset['val'] = {
    'data' : val_data, 
    'labels' : val_labels
}
dataset['split'] = n
dataset['permutation'] = P

torch.save(dataset, 'cifar100_validation_split.pth')