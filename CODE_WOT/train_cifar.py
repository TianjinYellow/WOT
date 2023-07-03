import argparse
import logging
import sys
import time
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import copy
import os

from wideresnet import WideResNet
from preactresnet import PreActResNet18

from utils import *



args = get_args()


if args.dataset=='cifar10':
    mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
    std = torch.tensor(cifar10_std).view(3,1,1).cuda()
elif args.dataset=='cifar100':
    mu = torch.tensor(CIFAR100_MEAN).view(3,1,1).cuda()
    std = torch.tensor(CIFAR100_STD).view(3,1,1).cuda()

def normalize(X):
    return (X - mu)/std

upper_limit, lower_limit = 1,0



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

def bn_update(loader, model,steps):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.
        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    for i,batch in enumerate(loader):
        if i>steps:
            break
        input,_=batch['input'], batch['target']
        input = input.cuda()
        b = input.data.size(0)

        momentum = b / (n + b)
        for module in momenta.keys():
            module.momentum = momentum

        model(normalize(input))
        n += b

    model.apply(lambda module: _set_momenta(module, momenta))



def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


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
        return ({'input': x.to(device).float(), 'target': y.to(device).long()} for (x,y) in self.dataloader)

    def __len__(self):
        return len(self.dataloader)


def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts,
               norm, early_stop=False,
               mixup=False, y_a=None, y_b=None, lam=None):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for _ in range(restarts):
        delta = torch.zeros_like(X).cuda()
        if norm == "l_inf":
            delta.uniform_(-epsilon, epsilon)
        elif norm == "l_2":
            delta.normal_()
            d_flat = delta.view(delta.size(0),-1)
            n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r/n*epsilon
        else:
            raise ValueError
        delta = clamp(delta, lower_limit-X, upper_limit-X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(normalize(X + delta))
            if early_stop:
                index = torch.where(output.max(1)[1] == y)[0]
            else:
                index = slice(None,None,None)
            if not isinstance(index, slice) and len(index) == 0:
                break
            if mixup:
                criterion = nn.CrossEntropyLoss()
                loss = mixup_criterion(criterion, model(normalize(X+delta)), y_a, y_b, lam)
            else:
                loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]
            if norm == "l_inf":
                d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
                scaled_g = g/(g_norm + 1e-10)
                d = (d + scaled_g*alpha).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()
        if mixup:
            criterion = nn.CrossEntropyLoss(reduction='none')
            all_loss = mixup_criterion(criterion, model(normalize(X+delta)), y_a, y_b, lam)
        else:
            all_loss = F.cross_entropy(model(normalize(X+delta)), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def update_parameters(model_parameters,parameters_origin,delta_p_all,weights,model,dicts):
    #assert len(delta_p_all)==10
    model_parameters=list(model_parameters)
    parameters_origin=list(parameters_origin)
    for i, (k,v) in enumerate(model.named_parameters()):
    #for i,(p,q) in enumerate(zip(parameters_origin,model_parameters)):
        p=parameters_origin[i]
        q=model_parameters[i]
        temp=0
        for j,delta in enumerate(delta_p_all):
            temp+=delta[i].data*weights[j][dicts[k]]
        q.data=p.data+temp.detach()

def update_weights(grad,delta_p_all,weights,model,dicts,lr=0.01,momentum_buffer=0,momentum=False,alpha=0.9):
    for i,(key,v) in enumerate(model.named_parameters()):
        g=grad[i]
        #g=g/(g.norm()+1e-12)
        for j,delta in enumerate(delta_p_all):    
            grad_weight_j=torch.sum(delta[i]*g)
            if momentum:
                momentum_buffer[j][dicts[key]]=momentum_buffer[j][dicts[key]]*alpha+grad_weight_j
            else:
                momentum_buffer[j][dicts[key]]=grad_weight_j
                
            weights[j][dicts[key]]=weights[j][dicts[key]]-momentum_buffer[j][dicts[key]]*lr
    weights=torch.clamp(weights,0,1.0)
    return weights

def update_parameters_two(parameters,parameters1):
    for (p,q) in zip(parameters,parameters1):
        p.data=q.data
def model_difference(p1,p2):
    diff=0.0
    for (p,q) in zip(p1,p2):
        diff+=(p-q).sum().item()
    print("difference",diff)
        
def main():
    if not os.path.exists(args.fname):
        os.makedirs(args.fname)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(args.fname, 'eval.log' if args.eval else 'output.log')),
            logging.StreamHandler()
        ])

    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    transforms = [Crop(32, 32), FlipLR()]
    if args.cutout:
        transforms.append(Cutout(args.cutout_len, args.cutout_len))
    if args.val:
        try:
            if args.dataset=='cifar10':
                dataset = torch.load("cifar10_validation_split.pth")
            elif args.dataset=='cifar100':
                dataset=torch.load("cifar100_validation_split.pth")
        except:
            print("Couldn't find a dataset with a validation split, did you run "
                  "generate_validation.py?")
            return
        val_set = list(zip(transpose(dataset['val']['data']/255.), dataset['val']['labels']))
        val_batches = Batches(val_set, args.batch_size, shuffle=False, num_workers=2)
    else:
        if args.dataset=='cifar10':
            dataset = cifar10(args.data_dir)
        elif args.dataset=='cifar100':
            dataset=cifar100(args.data_dir)           
    train_set = list(zip(transpose(pad(dataset['train']['data'], 4)/255.),
        dataset['train']['labels']))
    train_set_x = Transform(train_set, transforms)
    train_batches = Batches(train_set_x, args.batch_size, shuffle=True, set_random_choices=True, num_workers=2)

    test_set = list(zip(transpose(dataset['test']['data']/255.), dataset['test']['labels']))
    test_batches = Batches(test_set, args.batch_size, shuffle=False, num_workers=2)

    epsilon = (args.epsilon / 255.)
    pgd_alpha = (args.pgd_alpha / 255.)
    
    if args.dataset=='cifar100':
        num_classes=100
    elif args.dataset=='cifar10':
        num_classes=10

    if args.model == 'PreActResNet18':
        model = PreActResNet18(num_classes=num_classes)
    elif args.model == 'WideResNet':
        model = WideResNet(34,num_classes=num_classes, widen_factor=args.width_factor, dropRate=0.0)
    else:
        raise ValueError("Unknown model")

    model = nn.DataParallel(model).cuda()
    model_exploit=copy.deepcopy(model)
    model.train()

    if args.l2:
        decay, no_decay = [], []
        for name,param in model.named_parameters():
            if 'bn' not in name and 'bias' not in name:
                decay.append(param)
            else:
                no_decay.append(param)
        params = [{'params':decay, 'weight_decay':args.l2},
                  {'params':no_decay, 'weight_decay': 0 }]
    else:
        params = model.parameters()

    opt = torch.optim.SGD(params, lr=args.lr_max, momentum=0.9, weight_decay=5e-4)

    criterion = nn.CrossEntropyLoss()

    if args.attack == 'free':
        delta = torch.zeros(args.batch_size, 3, 32, 32).cuda()
        delta.requires_grad = True
    elif args.attack == 'fgsm' and args.fgsm_init == 'previous':
        delta = torch.zeros(args.batch_size, 3, 32, 32).cuda()
        delta.requires_grad = True

    if args.attack == 'free':
        epochs = int(math.ceil(args.epochs / args.attack_iters))
    else:
        epochs = args.epochs

    if args.lr_schedule == 'superconverge':
        lr_schedule = lambda t: np.interp([t], [0, args.epochs * 2 // 5, args.epochs], [0, args.lr_max, 0])[0]
    elif args.lr_schedule == 'piecewise':
        def lr_schedule(t):
            if t / args.epochs < 0.5:
                return args.lr_max
            elif t / args.epochs < 0.75:
                return args.lr_max / 10.
            else:
                return args.lr_max / 100.
    elif args.lr_schedule == 'linear':
        lr_schedule = lambda t: np.interp([t], [0, args.epochs // 3, args.epochs * 2 // 3, args.epochs], [args.lr_max, args.lr_max, args.lr_max / 10, args.lr_max / 100])[0]
    elif args.lr_schedule == 'onedrop':
        def lr_schedule(t):
            if t < args.lr_drop_epoch:
                return args.lr_max
            else:
                return args.lr_one_drop
    elif args.lr_schedule == 'multipledecay':
        def lr_schedule(t):
            return args.lr_max - (t//(args.epochs//10))*(args.lr_max/10)
    elif args.lr_schedule == 'cosine': 
        def lr_schedule(t): 
            return args.lr_max * 0.5 * (1 + np.cos(t / args.epochs * np.pi))


    best_test_robust_acc = 0
    best_val_robust_acc = 0
    if args.resume:
        start_epoch = args.resume
        model.load_state_dict(torch.load(os.path.join(args.fname, f'model_{start_epoch-1}.pth')))
        opt.load_state_dict(torch.load(os.path.join(args.fname, f'opt_{start_epoch-1}.pth')))
        logger.info(f'Resuming at epoch {start_epoch}')

        #best_test_robust_acc = torch.load(os.path.join(args.fname, f'model_best.pth'))['test_robust_acc']
        #if args.val:
        #    best_val_robust_acc = torch.load(os.path.join(args.fname, f'model_val.pth'))['val_robust_acc']
    else:
        start_epoch = 0

    if args.eval:
        if not args.resume:
            logger.info("No model loaded to evaluate, specify with --resume FNAME")
            return
        logger.info("[Evaluation mode]")

    logger.info('Epoch \t Train Time \t Test Time \t LR \t \t Train Loss \t Train Acc \t Train Robust Loss \t Train Robust Acc \t Test Loss \t Test Acc \t Test Robust Loss \t Test Robust Acc')
    steps=0.0
    parameters_origin=None
    delta_p_all=[]
    #dicts={}
    dicts=get_dicts(args,model)

    print("dicts",dicts)
    best_meta_acc=None
    gap=args.gap
    num_models=args.num_gaps
    flag=True
    for epoch in range(start_epoch, epochs):
        model.train()
        start_time = time.time()
        train_loss = 0
        train_acc = 0
        train_robust_loss = 0
        train_robust_acc = 0
        train_n = 0


        print("gap",gap)
        for i, batch in enumerate(train_batches):
            if args.eval:
                break
            X, y = batch['input'], batch['target']
            if args.mixup:
                X, y_a, y_b, lam = mixup_data(X, y, args.mixup_alpha)
                X, y_a, y_b = map(Variable, (X, y_a, y_b))
            lr = lr_schedule(epoch + (i + 1) / len(train_batches))
            opt.param_groups[0].update(lr=lr)

            if args.attack == 'pgd':
                # Random initialization
                if args.mixup:
                    delta = attack_pgd(model, X, y, epsilon, pgd_alpha, args.attack_iters, args.restarts, args.norm, mixup=True, y_a=y_a, y_b=y_b, lam=lam)
                else:
                    delta = attack_pgd(model, X, y, epsilon, pgd_alpha, args.attack_iters, args.restarts, args.norm)
                delta = delta.detach()
            elif args.attack == 'fgsm':
                delta = attack_pgd(model, X, y, epsilon, args.fgsm_alpha*epsilon, 1, 1, args.norm)
            # Standard training
            elif args.attack == 'none':
                delta = torch.zeros_like(X)

            robust_output = model(normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)))
            if args.mixup:
                robust_loss = mixup_criterion(criterion, robust_output, y_a, y_b, lam)
            else:
                robust_loss = criterion(robust_output, y)

            if args.l1:
                for name,param in model.named_parameters():
                    if 'bn' not in name and 'bias' not in name:
                        robust_loss += args.l1*param.abs().sum()

            opt.zero_grad()
            robust_loss.backward()
            opt.step()
            output = model(normalize(X))
            if args.mixup:
                loss = mixup_criterion(criterion, output, y_a, y_b, lam)
            else:
                loss = criterion(output, y)

            train_robust_loss += robust_loss.item() * y.size(0)
            train_robust_acc += (robust_output.max(1)[1] == y).sum().item()
            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)
 

            if epoch>=args.MetaStartEpoch:
                steps+=1
                if parameters_origin is None:
                    parameters_origin=copy.deepcopy(list(model.parameters()))                                   
                    parameters_pre=copy.deepcopy(list(model.parameters()))
                    model_exploit.load_state_dict(copy.deepcopy(model.state_dict()))
                else:
                    if steps%gap==0:
                        delta_p_temp=[]
                        for p_pre,p_cur in zip(parameters_pre,model.parameters()):
                            delta_p=(p_cur-p_pre).detach()
                            delta_p_temp.append(delta_p)
                        parameters_pre=copy.deepcopy(list(model.parameters()))       
                        delta_p_all.append(delta_p_temp)
                        
                    if steps%(gap*num_models)==0:   #######optmizing  
                        if args.val:
                            model.eval()
                            val_loss = 0
                            val_acc = 0
                            val_robust_loss = 0
                            val_robust_acc = 0
                            val_n = 0
                            for i, batch in enumerate(val_batches):
                                X, y = batch['input'], batch['target']

                                # Random initialization
                                if args.attack == 'none':
                                    delta = torch.zeros_like(X)
                                else:
                                    delta = attack_pgd(model, X, y, epsilon, pgd_alpha, args.attack_iters, args.restarts, args.norm, early_stop=args.eval)
                                delta = delta.detach()

                                robust_output = model(normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)))
                                robust_loss = criterion(robust_output, y)

                                output = model(normalize(X))
                                loss = criterion(output, y)

                                val_robust_loss += robust_loss.item() * y.size(0)
                                val_robust_acc += (robust_output.max(1)[1] == y).sum().item()
                                val_loss += loss.item() * y.size(0)
                                val_acc += (output.max(1)[1] == y).sum().item()
                                val_n += y.size(0)
                            print("val acc",val_acc/val_n) 

                        
                        #variables=[torch.ones(1,device=device) for i in range(10)]
                        #variables=torch.zeros((num_models,args.layer_wise),device=device)
                        if args.initialize_type=='random':
                            variables=torch.rand((num_models,args.layer_wise),device=device)
                        elif args.initialize_type=='zero':
                            variables=torch.zeros((num_models,args.layer_wise),device=device)
                        elif args.initialize_type=='one':
                            variables=torch.ones((num_models,args.layer_wise),device=device)
                        
                        model_exploit.load_state_dict(copy.deepcopy(model.state_dict()))
                        #print("before")
                        #model_difference(model.parameters(),model_exploit.parameters())
                        update_parameters(model_exploit.parameters(),parameters_origin,delta_p_all,variables,model,dicts)
                        #print("after")
                        ####################################Check the difference##################################
#                         for i,(p,q) in enumerate(zip(model_exploit.parameters(),model.parameters())):
#                             print("")                        
                        ###################################Check End##############################################
                        delta=0.0
                        if epoch>=args.train_mode_epoch:
                            model_exploit.train()
                        else:
                            model_exploit.eval()
                        print("#####################Optimize weights###################################")
                        for i in range(10):
                            val_n=0
                            val_robust_acc=0.0
                            val_robust_loss=0.0
                            momentum_buffer=torch.zeros_like(variables)
                            #bn_update(train_batches,model_exploit,20)
                            for j, batch in enumerate(val_batches):
                                X, y = batch['input'], batch['target']
                                # Random initialization
                                if args.attack == 'none':
                                    delta = torch.zeros_like(X)
                                else:#
                                    delta = attack_pgd(model_exploit, X, y, epsilon, pgd_alpha, args.attack_iters, args.restarts, args.norm, early_stop=args.eval)
                                delta = delta.detach()
                                
                                robust_output = model_exploit(normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)))
                                robust_loss = criterion(robust_output, y)
                                grad=torch.autograd.grad(robust_loss,model_exploit.parameters())
                                variables=update_weights(grad,delta_p_all,variables,model,dicts,lr=0.1,momentum_buffer=momentum_buffer,momentum=True,alpha=0.9)                                    
                                update_parameters(model_exploit.parameters(),parameters_origin,delta_p_all,variables,model,dicts)
                                val_robust_loss += robust_loss.item() * y.size(0)
                                val_robust_acc += (robust_output.max(1)[1] == y).sum().item()
                                val_n += y.size(0)
                            print("i",i,"validate robust acc:",val_robust_acc/val_n,"loss",val_robust_loss/val_n)
                        delta_p_all=[]
                        bn_update(train_batches,model_exploit,400)
                        print("variables:",variables)
                        
                        #model.load_state_dict(copy.deepcopy(model_exploit.state_dict()))
                        print("#####################Optimize weights End###################################")
                        if epoch>=150 and flag: 
                            gap=gap*args.times
                            flag=False
                        model.eval()
                        model_exploit.eval()
                        test_loss = 0
                        test_acc = 0
                        test_robust_loss = 0
                        test_robust_acc = 0
                        test_n = 0
                        test_acc_orgin=0.0
                        for i, batch in enumerate(test_batches):
                            X, y = batch['input'], batch['target']

                            # Random initialization
                            if args.attack == 'none':
                                delta = torch.zeros_like(X)
                            else:
                                delta = attack_pgd(model_exploit, X, y, epsilon, pgd_alpha, args.attack_iters, args.restarts, args.norm, early_stop=args.eval)
                            delta = delta.detach()

                            robust_output = model(normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)))
                            out=model_exploit(normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)))
                            robust_loss = criterion(robust_output, y)

                            output = model_exploit(normalize(X))
                            loss = criterion(output, y)

                            test_robust_loss += robust_loss.item() * y.size(0)
                            test_robust_acc += (robust_output.max(1)[1] == y).sum().item()

                            test_acc_orgin+=(out.max(1)[1]==y).sum().item()

                            test_loss += loss.item() * y.size(0)
                            test_acc += (output.max(1)[1] == y).sum().item()
                            test_n += y.size(0)
                        print("meta adapted robust acc:", test_acc_orgin/test_n,"orginal robust acc:",test_robust_acc/test_n,"meta adapted clean acc:",test_acc/test_n)
                        if best_meta_acc is None:
                            best_meta_acc=test_acc_orgin/test_n
                        else:
                            if best_meta_acc<(test_acc_orgin/test_n):
                                best_meta_acc=test_acc_orgin/test_n
                                print("save best meta model at ", epoch )
                                if args.file_name is None:
                                    torch.save(model_exploit.state_dict(),"./train_models/best_meta_reinit_"+str(args.reinitialize)+"_initType_"+str(args.initialize_type)+"_"+str(args.model)+"_"+args.dataset+"_gap_"+str(args.gap)+"_numgaps_"+str(args.num_gaps)+"_trainmodeepoch_"+str(args.train_mode_epoch)+"_momentum09_layerwise_"+str(args.layer_wise)+"_repeat_"+str(args.repeat)+"_MetaStart_"+str(args.MetaStartEpoch)+"_times_"+str(args.times)+".pt")
                                else:
                                     torch.save(model_exploit.state_dict(),"./train_models/best_meta_"+args.file_name+".pt")
                        if args.file_name is None:             
                            torch.save(model_exploit.state_dict(),"./train_models/last_meta_reinit_"+str(args.reinitialize)+"_initType_"+str(args.initialize_type)+"_"+str(args.model)+"_"+args.dataset+"_gap_"+str(args.gap)+"_numgaps_"+str(args.num_gaps)+"_trainmodeepoch_"+str(args.train_mode_epoch)+"_momentum09_layerwise_"+str(args.layer_wise)+"_repeat_"+str(args.repeat)+"_MetaStart_"+str(args.MetaStartEpoch)+"_times_"+str(args.times)+".pt")
                        else:
                            torch.save(model_exploit.state_dict(),"./train_models/last_meta_"+args.file_name+".pt")
                        #torch.save(model_exploit.state_dict(),"./checkpoints1/"+str(epoch)+"_last_meta_reinit_"+str(args.reinitialize)+"_initType_"+str(args.initialize_type)+"_"+str(args.model)+"_"+args.dataset+"_gap_"+str(args.gap)+"_numgaps_"+str(args.num_gaps)+"_trainmodeepoch_"+str(args.train_mode_epoch)+"_momentum09_layerwise_"+str(args.layer_wise)+"_repeat_"+str(args.repeat)+"_MetaStart_"+str(args.MetaStartEpoch)+"_times_"+str(args.times)+".pt")    
                        model.load_state_dict(copy.deepcopy(model_exploit.state_dict()))
                        parameters_origin=copy.deepcopy(list(model_exploit.parameters()))
                        parameters_pre=copy.deepcopy(list(model_exploit.parameters()))
                        model.train()
                        ###remove momentum buffer
                        opt_temp = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=0.9, weight_decay=5e-4)
                        if args.reinitialize==1:
                            opt=opt_temp
                        else:
                            for i,(p,q) in enumerate(zip(opt.param_groups[0]['params'],opt_temp.param_groups[0]['params'])):
                                state=opt.state[p]
                                opt_temp.state[q]['momentum_buffer']=state['momentum_buffer']
                            opt=opt_temp
        train_time = time.time()

        model.eval()
        #model_exploit.eval()
        test_loss = 0
        test_acc = 0
        test_robust_loss = 0
        test_robust_acc = 0
        test_n = 0
        test_acc_orgin=0.0
        for i, batch in enumerate(test_batches):
            X, y = batch['input'], batch['target']

            # Random initialization
            if args.attack == 'none':
                delta = torch.zeros_like(X)
            else:
                delta = attack_pgd(model, X, y, epsilon, pgd_alpha, args.attack_iters, args.restarts, args.norm, early_stop=args.eval)
            delta = delta.detach()

            robust_output = model(normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)))
            #out=model_exploit(normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)))
            
            robust_loss = criterion(robust_output, y)

            output = model(normalize(X))
            loss = criterion(output, y)

            test_robust_loss += robust_loss.item() * y.size(0)
            test_robust_acc += (robust_output.max(1)[1] == y).sum().item()
            
            #test_acc_orgin+=(out.max(1)[1]==y).sum().item()
            
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            test_n += y.size(0)
        #if epoch>=10 and steps%400==0:
        #    print("meta test acc", test_acc_orgin/test_n,"orginal adapted",test_acc/test_n)
        test_time = time.time()

        if args.val:
            val_loss = 0
            val_acc = 0
            val_robust_loss = 0
            val_robust_acc = 0
            val_n = 0
            for i, batch in enumerate(val_batches):
                X, y = batch['input'], batch['target']

                # Random initialization
                if args.attack == 'none':
                    delta = torch.zeros_like(X)
                else:
                    delta = attack_pgd(model, X, y, epsilon, pgd_alpha, args.attack_iters, args.restarts, args.norm, early_stop=args.eval)
                delta = delta.detach()

                robust_output = model(normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)))
                robust_loss = criterion(robust_output, y)

                output = model(normalize(X))
                loss = criterion(output, y)

                val_robust_loss += robust_loss.item() * y.size(0)
                val_robust_acc += (robust_output.max(1)[1] == y).sum().item()
                val_loss += loss.item() * y.size(0)
                val_acc += (output.max(1)[1] == y).sum().item()
                val_n += y.size(0)

        if not args.eval:
            logger.info('%d \t %.1f \t \t %.1f \t \t %.4f \t %.4f \t %.4f \t %.4f \t \t %.4f \t \t %.4f \t %.4f \t %.4f \t \t %.4f',
                epoch, train_time - start_time, test_time - train_time, lr,
                train_loss/train_n, train_acc/train_n, train_robust_loss/train_n, train_robust_acc/train_n,
                test_loss/test_n, test_acc/test_n, test_robust_loss/test_n, test_robust_acc/test_n)

            if args.val:
                logger.info('validation %.4f \t %.4f \t %.4f \t %.4f',
                    val_loss/val_n, val_acc/val_n, val_robust_loss/val_n, val_robust_acc/val_n)

                # if val_robust_acc/val_n > best_val_robust_acc:
                #     torch.save({
                #             'state_dict':model.state_dict(),
                #             'test_robust_acc':test_robust_acc/test_n,
                #             'test_robust_loss':test_robust_loss/test_n,
                #             'test_loss':test_loss/test_n,
                #             'test_acc':test_acc/test_n,
                #             'val_robust_acc':val_robust_acc/val_n,
                #             'val_robust_loss':val_robust_loss/val_n,
                #             'val_loss':val_loss/val_n,
                #             'val_acc':val_acc/val_n,
                #         }, os.path.join(args.fname, f'model_val.pth'))
                #     best_val_robust_acc = val_robust_acc/val_n

            # save checkpoint
            #if epoch <=args.MetaStartEpoch:
            ##    if (epoch+1) % 10 == 0 or epoch+1 == epochs:
            #        torch.save(model.state_dict(),"./checkpoints1/"+str(epoch)+"_last_meta_reinit_"+str(args.reinitialize)+"_initType_"+str(args.initialize_type)+"_"+str(args.model)+"_"+args.dataset+"_gap_"+str(args.gap)+"_numgaps_"+str(args.num_gaps)+"_trainmodeepoch_"+str(args.train_mode_epoch)+"_momentum09_layerwise_"+str(args.layer_wise)+"_repeat_"+str(args.repeat)+"_MetaStart_"+str(args.MetaStartEpoch)+"_times_"+str(args.times)+".pt")
            #    torch.save(model.state_dict(), os.path.join(args.fname, f'model_{epoch}.pth'))
            #    torch.save(opt.state_dict(), os.path.join(args.fname, f'opt_{epoch}.pth'))

            # save best
            # if test_robust_acc/test_n > best_test_robust_acc:
            #     torch.save({
            #             'state_dict':model.state_dict(),
            #             'test_robust_acc':test_robust_acc/test_n,
            #             'test_robust_loss':test_robust_loss/test_n,
            #             'test_loss':test_loss/test_n,
            #             'test_acc':test_acc/test_n,
            #         }, os.path.join(args.fname, f'model_best_meta.pth'))
            #     best_test_robust_acc = test_robust_acc/test_n
        else:
            logger.info('%d \t %.1f \t \t %.1f \t \t %.4f \t %.4f \t %.4f \t %.4f \t \t %.4f \t \t %.4f \t %.4f \t %.4f \t \t %.4f',
                epoch, train_time - start_time, test_time - train_time, -1,
                -1, -1, -1, -1,
                test_loss/test_n, test_acc/test_n, test_robust_loss/test_n, test_robust_acc/test_n)
            return


if __name__ == "__main__":
    main()
