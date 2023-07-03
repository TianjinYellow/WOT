import numpy as np
from collections import namedtuple
import torch
from torch import nn
import torchvision
import argparse
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

################################################################
## Components from https://github.com/davidcpage/cifar10-fast ##
################################################################

#####################
## data preprocessing
#####################

cifar10_mean = (0.4914, 0.4822, 0.4465) # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar10_std = (0.2471, 0.2435, 0.2616) # equals np.std(train_set.train_data, axis=(0,1,2))/255


mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
std = torch.tensor(cifar10_std).view(3,1,1).cuda()
CIFAR100_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

#def normalize(X):
#    return (X - mu)/std

def pad(x, border=4):
    return np.pad(x, [(0, 0), (border, border), (border, border), (0, 0)], mode='reflect')

def transpose(x, source='NHWC', target='NCHW'):
    return x.transpose([source.index(d) for d in target]) 

#####################
## data augmentation
#####################

class Crop(namedtuple('Crop', ('h', 'w'))):
    def __call__(self, x, x0, y0):
        return x[:,y0:y0+self.h,x0:x0+self.w]

    def options(self, x_shape):
        C, H, W = x_shape
        return {'x0': range(W+1-self.w), 'y0': range(H+1-self.h)}
    
    def output_shape(self, x_shape):
        C, H, W = x_shape
        return (C, self.h, self.w)
    
class FlipLR(namedtuple('FlipLR', ())):
    def __call__(self, x, choice):
        return x[:, :, ::-1].copy() if choice else x 
        
    def options(self, x_shape):
        return {'choice': [True, False]}

class Cutout(namedtuple('Cutout', ('h', 'w'))):
    def __call__(self, x, x0, y0):
        x = x.copy()
        x[:,y0:y0+self.h,x0:x0+self.w].fill(0.0)
        return x

    def options(self, x_shape):
        C, H, W = x_shape
        return {'x0': range(W+1-self.w), 'y0': range(H+1-self.h)} 
    
    
class Transform():
    def __init__(self, dataset, transforms):
        self.dataset, self.transforms = dataset, transforms
        self.choices = None
        
    def __len__(self):
        return len(self.dataset)
           
    def __getitem__(self, index):
        data, labels = self.dataset[index]
        for choices, f in zip(self.choices, self.transforms):
            args = {k: v[index] for (k,v) in choices.items()}
            data = f(data, **args)
        return data, labels
    
    def set_random_choices(self):
        self.choices = []
        x_shape = self.dataset[0][0].shape
        N = len(self)
        for t in self.transforms:
            options = t.options(x_shape)
            x_shape = t.output_shape(x_shape) if hasattr(t, 'output_shape') else x_shape
            self.choices.append({k:np.random.choice(v, size=N) for (k,v) in options.items()})

#####################
## dataset
#####################

def cifar10(root):
    train_set = torchvision.datasets.CIFAR10(root=root, train=True, download=True)
    test_set = torchvision.datasets.CIFAR10(root=root, train=False, download=True)
    return {
        'train': {'data': train_set.data, 'labels': train_set.targets},
        'test': {'data': test_set.data, 'labels': test_set.targets}
    }

def cifar100(root):
    train_set = torchvision.datasets.CIFAR100(root=root, train=True, download=True)
    test_set = torchvision.datasets.CIFAR100(root=root, train=False, download=True)
    return {
        'train': {'data': train_set.data, 'labels': train_set.targets},
        'test': {'data': test_set.data, 'labels': test_set.targets}
    }


#####################
## data loading
#####################

class Batches():
    def __init__(self, dataset, batch_size, shuffle, set_random_choices=False, num_workers=0, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.set_random_choices = set_random_choices
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=shuffle, drop_last=drop_last
        )
    
    def __iter__(self):
        if self.set_random_choices:
            self.dataset.set_random_choices() 
        return ({'input': x.to(device).half(), 'target': y.to(device).long()} for (x,y) in self.dataloader)
    
    def __len__(self): 
        return len(self.dataloader)




def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True

def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]

def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)

def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum

def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='PreActResNet18')
    parser.add_argument('--l2', default=0, type=float)
    parser.add_argument('--l1', default=0, type=float)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='../cifar-data', type=str)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr-schedule', default='piecewise', choices=['superconverge', 'piecewise', 'linear', 'piecewisesmoothed', 'piecewisezoom', 'onedrop', 'multipledecay', 'cosine'])
    parser.add_argument('--lr-max', default=0.1, type=float)
    parser.add_argument('--lr-one-drop', default=0.01, type=float)
    parser.add_argument('--lr-drop-epoch', default=100, type=int)
    parser.add_argument('--attack', default='pgd', type=str, choices=['pgd', 'fgsm', 'free', 'none'])
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--attack-iters', default=10, type=int)
    parser.add_argument('--restarts', default=1, type=int)
    parser.add_argument('--pgd-alpha', default=2, type=float)
    parser.add_argument('--fgsm-alpha', default=1.25, type=float)
    parser.add_argument('--norm', default='l_inf', type=str, choices=['l_inf', 'l_2'])
    parser.add_argument('--fgsm-init', default='random', choices=['zero', 'random', 'previous'])
    parser.add_argument('--fname', default='cifar_model', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--width-factor', default=10, type=int)
    parser.add_argument('--resume', default=0, type=int)
    parser.add_argument('--cutout', action='store_true')
    parser.add_argument('--cutout-len', type=int)
    parser.add_argument('--mixup', action='store_true')
    parser.add_argument('--mixup-alpha', type=float)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--chkpt-iters', default=10, type=int)
    ###########ours##############################################################################
    parser.add_argument('--reinitialize', default=0, type=int)
    parser.add_argument('--initialize_type',default='zero',type=str,choices=['zero','one','random'])   
    parser.add_argument('--gap',default=100,type=int)
    parser.add_argument('--num_gaps',default=4,type=int)
    parser.add_argument('--layer_wise',default=1,type=int)
    parser.add_argument('--MetaStartEpoch',default=50,type=int)
    parser.add_argument('--repeat',default=0,type=int)
    parser.add_argument('--file_name', default=None, type=str)
    parser.add_argument('--dataset', default='cifar10', type=str)
    parser.add_argument('--train_mode_epoch',default=150,type=int)
    parser.add_argument('--times',default=2,type=int)
    parser.add_argument('--meta_loss',default='CE',choices=['kl','CE'])
    return parser.parse_args()


def get_dicts(args,model):
    dicts={}
    if args.model=='PreActResNet18':
        if args.layer_wise==2:
            for  k,v in model.named_parameters():
                if "module.conv1" in k:
                    dicts[k]=0
                elif "layer1" in k:
                    dicts[k]=0
                elif "layer2" in k:
                    dicts[k]=0
                elif "layer3" in k:
                    dicts[k]=0
                elif "layer4" in k:
                    dicts[k]=0
                else:
                    dicts[k]=1
        elif args.layer_wise==3:
            for  k,v in model.named_parameters():
                if "module.conv1" in k:
                    dicts[k]=0
                elif "layer1" in k:
                    dicts[k]=1
                elif "layer2" in k:
                    dicts[k]=1
                elif "layer3" in k:
                    dicts[k]=1
                elif "layer4" in k:
                    dicts[k]=1
                else:
                    dicts[k]=2
        elif args.layer_wise==4:
            for  k,v in model.named_parameters():
                if "module.conv1" in k:
                    dicts[k]=0
                elif "layer1" in k:
                    dicts[k]=1
                elif "layer2" in k:
                    dicts[k]=1
                elif "layer3" in k:
                    dicts[k]=2
                elif "layer4" in k:
                    dicts[k]=2
                else:
                    dicts[k]=3
        elif args.layer_wise==5:
            for  k,v in model.named_parameters():
                if "module.conv1" in k:
                    dicts[k]=0
                elif "layer1" in k:
                    dicts[k]=0
                elif "layer2" in k:
                    dicts[k]=1
                elif "layer3" in k:
                    dicts[k]=2
                elif "layer4" in k:
                    dicts[k]=3
                else:
                    dicts[k]=4
        elif args.layer_wise==1:
            for  k,v in model.named_parameters():
                if "module.conv1" in k:
                    dicts[k]=0
                elif "layer1" in k:
                    dicts[k]=0
                elif "layer2" in k:
                    dicts[k]=0
                elif "layer3" in k:
                    dicts[k]=0
                elif "layer4" in k:
                    dicts[k]=0
                else:
                    dicts[k]=0
        elif args.layer_wise==6:
            for  k,v in model.named_parameters():
                if "module.conv1" in k:
                    dicts[k]=0
                elif "layer1" in k:
                    dicts[k]=1
                elif "layer2" in k:
                    dicts[k]=2
                elif "layer3" in k:
                    dicts[k]=3
                elif "layer4" in k:
                    dicts[k]=4
                else:
                    dicts[k]=5
    elif args.model=='WideResNet':
        if args.layer_wise==1:
            for k,v in model.named_parameters():
                if "module.conv1" in k:
                    dicts[k]=0
                elif "block1" in k:
                    dicts[k]=0
                elif "block2" in k:
                    dicts[k]=0
                elif "block3" in k:
                    dicts[k]=0
                else:
                    dicts[k]=0
        elif args.layer_wise==2:
            for k,v in model.named_parameters():
                if "module.conv1" in k:
                    dicts[k]=0
                elif "block1" in k:
                    dicts[k]=0
                elif "block2" in k:
                    dicts[k]=0
                elif "block3" in k:
                    dicts[k]=0
                else:
                    dicts[k]=1
        elif args.layer_wise==3:
            for k,v in model.named_parameters():
                if "module.conv1" in k:
                    dicts[k]=0
                elif "block1" in k:
                    dicts[k]=0
                elif "block2" in k:
                    dicts[k]=1
                elif "block3" in k:
                    dicts[k]=1
                else:
                    dicts[k]=2
        elif args.layer_wise==4:
            for k,v in model.named_parameters():
                if "module.conv1" in k:
                    dicts[k]=0
                elif "block1" in k:
                    dicts[k]=0
                elif "block2" in k:
                    dicts[k]=1
                elif "block3" in k:
                    dicts[k]=2
                else:
                    dicts[k]=3
        elif args.layer_wise==5:
            for k,v in model.named_parameters():
                if "module.conv1" in k:
                    dicts[k]=0
                elif "block1" in k:
                    dicts[k]=1
                elif "block2" in k:
                    dicts[k]=2
                elif "block3" in k:
                    dicts[k]=3
                else:
                    dicts[k]=4
    elif args.model=='vgg' or args.model=='vgg19':
        list_layers=[64,128,256,512,10]
        dicts_id={64:0,128:1,256:2,512:3,10:4}
        if args.layer_wise==1:
            for k,v in model.named_parameters():
                dicts[k]=0
        elif args.layer_wise==5:
            for k,v in model.named_parameters():
                #if "features" in k:
                if v.shape[0] in list_layers:
                    dicts[k]=dicts_id[v.shape[0]]
                else:
                    assert False, "layer id not correct!"
                #else:
                #    dicts[k]=1
    return dicts
