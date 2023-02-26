import os
import copy
import random
import numpy as np
import pandas as pd
from collections import OrderedDict
import argparse

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.models
import torchvision.transforms

# Self-made libraries
from dataset import make_loader
from Noise_Adapt_Master.utils import get_device, seed_torch, accuracy, AverageMeter, get_T, load_model, lr_reduce, fwd_loss
from Noise_Adapt_Master.estimation import estimator, estimator_multi


# Overall settings
datapath = './data/'
dataset_names = ['FashionMNIST0.5.npz', 'FashionMNIST0.6.npz', 'CIFAR.npz']
model_names_MNIST = ['CNN', 'resnet18']
model_names_CIFAR = ['CNN', 'resnet20']
resultroot = './results/'
if not os.path.exists(resultroot):
    os.makedirs(resultroot)


# Argument parser for CLI interaction
parser = argparse.ArgumentParser(description = "Run experiments on various hyperspectral datasets")

# Dataset options
group_dataset = parser.add_argument_group('Dataset')
group_dataset.add_argument('--train_sample', type = float, default = 0.8, help = "Percentage of samples to use for training (default: 80%)")
group_dataset.add_argument('--val_sample', type = float, default = 0.2, help = "Percentage of samples to use for validation (default: 20%)")
group_dataset.add_argument('--flip_augmentation', type = bool, default = True, help = 'Whether to randomly flip the data')
group_dataset.add_argument('--num_workers', type = int, default = 8, help = "Number of workers of torch Dataloader.")
group_dataset.add_argument('--dataset', type = int, default = 1, help = "Number of the chosen dataset.")

# Training options
group_train = parser.add_argument_group('Training')
group_train.add_argument('--times', type = int, default = 1, help = "Times of running")
group_train.add_argument('--cuda', type = str, default = '0' if torch.cuda.is_available() else '-', help = "Specify CUDA device (set '-' to learn on CPU)")
group_train.add_argument('--model_name', type = str, default = 'CNN', help = "Choose a model by name")
group_train.add_argument('--epoch', type = int, default = 10, help = "Training epochs (optional, if absent will be set by the model)")
group_train.add_argument('--epoch_fw', type = int, default = 30, help = "Training epochs of forward learning (optional, if absent will be set by the model)")
group_train.add_argument('--lr', type = float, default = 1e-1, help = "Learning rate. Please don't set a small number if the learning rate reduction is on")
group_train.add_argument('--lr_rd', type = float, default = 1, help = "The step length of learning rate reduction. Set to be bigger than epoch if the reduction is not needed.")
group_train.add_argument('--batch_size', type = int, default = 1024, help = "Batch size (optional, if absent will be set by the model")
group_train.add_argument('--multi_anchor', type = bool, default = False, help = "Batch size (optional, if absent will be set by the model")

# Conclusion of args
args = parser.parse_args()
hyperparams = vars(args)
device = get_device(hyperparams['cuda'])
seed_torch(device, 1)
hyperparams['device'] = device
model_choice = hyperparams['model_name']



# Training step
def train(epoch, model, optimizer, loss_func, train_loader, device, T_matrix, is_fwd = False):
    """
    Training.
    """
    top1_acc_meter = AverageMeter()
    loss_meter = AverageMeter()

    model.train()
    for step, (data, label) in enumerate(train_loader):
        # Train
        data = data.to(device)
        label = label.to(device)
        preds = model(data)
        if is_fwd:
            # Adapting layer
            loss = fwd_loss(preds, label, T_matrix.to(device))
        else:
            loss = loss_func(preds, label)

        preds = F.softmax(preds, dim = 1)
        [top1_acc] = accuracy(preds.data, label.data, topk=(1,))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        min_batch_size = data.size(0)
        top1_acc_meter.update(top1_acc.cpu().item(), min_batch_size)
        loss_meter.update(loss.item(), min_batch_size)

    print("Train epoch:", epoch, "Accuracy:", top1_acc_meter.avg, 'Loss', loss.item())
    return loss.item()


# Testing step
def test(epoch, model, val_loader, device, T_matrix, is_val = True, is_fwd = False):
    """
    Validation and testing.
    """
    top1_acc_meter = AverageMeter()

    model.eval()
    for step, (data, label) in enumerate(val_loader):
        data = data.to(device)
        label = label.to(device)
        with torch.no_grad():
            preds = model(data)
        
        if is_fwd and is_val:
            # Adapting layer
            preds = preds @ T_matrix.to(device)
        preds = F.softmax(preds, dim = 1)
        [top1_acc] = accuracy(preds.data, label.data, topk=(1,))
        
        min_batch_size = data.size(0)
        top1_acc_meter.update(top1_acc.item(), min_batch_size)

    top1_acc_avg = top1_acc_meter.avg
    if is_val:
        print("Validate epoch ",epoch," Accuracy ",top1_acc_avg)
    else:
        print("Test epoch ",epoch," Accuracy ",top1_acc_avg)

    return top1_acc_avg


# Training process
def run(train_loader, val_loader, is_forward, **hyperparams):
    """
    Entire training process.
    """
    device = hyperparams['device']
    top1_acc_best = 0
    ACCs = []
    losses = []
    
    # Create output directory
    outdir = './' + hyperparams['model_name']
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    # Load model
    model = load_model(**hyperparams)
    if torch.cuda.device_count() > 1:
        print(torch.cuda.device_count(), "GPUs are used!")
        model = nn.DataParallel(model)
    model = model.to(device)

    # Criterion and optimizer
    loss_func = nn.CrossEntropyLoss()
    

    fwd_path = ''
    if is_forward:
        fwd_path = 'fwd_'

    best_model_path = outdir + '/' + fwd_path + hyperparams['model_name'] + '_best.pth'
    best_epoch = 0
    lr = hyperparams['lr']
    lr_rd = hyperparams['lr_rd']

    for epoch in range(1, hyperparams['epoch'] + 1):
        if (epoch - 1) % lr_rd == 0 and epoch != 1:
            lr = lr_reduce(lr, (epoch - 1) / lr_rd, hyperparams['epoch'])
        optimizer = torch.optim.SGD(
            params = model.parameters(),
            lr = lr,
            momentum = 0.9,
            weight_decay = 1e-5
        )
        T_matrix =  hyperparams['T']

        # Train and validatoin
        loss_ = train(epoch, model, optimizer, loss_func, train_loader, device, T_matrix, is_fwd = is_forward)
        top1_acc_avg = test(epoch, model, val_loader, device, T_matrix, is_fwd = is_forward)
        
        ACCs.append(top1_acc_avg)
        losses.append(loss_)
        # Saving best model
        if (top1_acc_avg > top1_acc_best):
            state = OrderedDict([
                ('config', hyperparams),
                ('state_dict', model.state_dict()),
                ('optimizer', optimizer.state_dict()),
                ('epoch', epoch),
                ('top1-accuracy', top1_acc_avg),
            ])
            
            torch.save(state, best_model_path)
            top1_acc_best = top1_acc_avg
            print('new_best top-1 ACC', top1_acc_best)
            best_epoch = epoch
    
    del(model)
    
    return best_model_path, ACCs, losses, best_epoch


# Main result of a single run
if __name__ == '__main__':
    print('Configures:', hyperparams)

    # Datasets
    model_names = model_names_MNIST
    if hyperparams['dataset'] == 2:
        model_names = model_names_CIFAR
    if model_choice != 'all':
        if hyperparams['model_name'] not in model_names:
            print(hyperparams['model_name'], 'is not suitable for this dataset, please choose another one.')
            exit(0)
        model_names = [hyperparams['model_name']]
    hyperparams['name'] = dataset_names[hyperparams['dataset']]
    resultdata = resultroot + hyperparams['name'] + '/'
    if not os.path.exists(resultdata):
        os.makedirs(resultdata)

    print('loading data', hyperparams['name'][:-4], '...')
    #load data
    dataset = np.load(datapath + hyperparams['name'])
    Xtrval = dataset['Xtr']
    Strval = dataset['Str']
    Xts = dataset['Xts']
    Yts = dataset['Yts']
    print(Yts.max() + 1, 'classes in the dataset')

    num_classes = Yts.max() + 1
    channels = Xtrval.shape[-1] if len(Xtrval.shape) == 4 else 1
    hyperparams['num_classes'] = num_classes
    hyperparams['channels'] = channels
    hyperparams['img_size'] = Xtrval.shape[1] * Xtrval.shape[2]
    hyperparams['img_shape'] = (Xtrval.shape[1] , Xtrval.shape[2])

    # Model
    print('Using model', hyperparams['model_name'], '...')
    resultpath = resultdata + hyperparams['model_name'] + '/'
    if not os.path.exists(resultpath):
        os.makedirs(resultpath)
    
    ACC_fwds = []
    ACC_ori = []
    T_errs = []
    
    for k in range(hyperparams['times']):
        print(k + 1, 'times of running ...')
        resultpath_times = resultpath + str(k) + '/'
        if not os.path.exists(resultpath_times):
            os.makedirs(resultpath_times)
        [train_loader, val_loader] = make_loader(Xtrval, Strval, True, **hyperparams)
        [test_loader] = make_loader(Xts, Yts, False, **hyperparams)


        # Origin
        print('Training with no T matrix ...')
        hyperparams['T'] = torch.eye(num_classes)
        t0 = time.time()
        hyperparams['best_path'], ACCs, losses, best_epoch = run(train_loader, val_loader, False, **hyperparams)
        t1 = time.time()
        print('train time', (t1 - t0)/hyperparams['epoch'])
        
        # Testing normal model
        model = load_model(**hyperparams)
        model.load_state_dict(torch.load(hyperparams['best_path'])['state_dict'])
        model = model.to(hyperparams['device'])
        print('Without T matrix ...')
        originbck_top1_acc = test('T', model, test_loader, device, T_matrix = torch.eye(num_classes), is_val = False, is_fwd = True)
        ACC_ori.append(originbck_top1_acc)
        del(model)
        with open(resultpath_times + 'Origin_Train.csv', 'w') as f:
            txt = 'Epochs,Acc,Loss'
            for i in range(len(ACCs)):
                txt += '\n' + str(i) + ',' + str(ACCs[i]) +',' + str(losses[i])
            f.write(txt)


        # Estimate
        print('Estimating T matrix ...')
        hyperparams['T'] = get_T(**hyperparams)
        if hyperparams['multi_anchor']:
            print('Multi-anchor estimation')
            T_est = estimator_multi(train_loader, load_model(**hyperparams).to(device), hyperparams['best_path'], device = device, num_classes = num_classes)
        else:
            print('Single-anchor estimation')
            T_est = estimator(train_loader, load_model(**hyperparams).to(device), hyperparams['best_path'], device = device, num_classes = num_classes)
        with open(resultpath_times + 'Tmat.csv', 'w') as f:
            txt = 'est'
            for i in range(num_classes):
                txt += '\n' + str(T_est[i, 0])
                for j in range(1, num_classes):
                    txt += ',' + str(T_est[i, j])
            txt += '\nori'
            for i in range(num_classes):
                txt += '\n' + str(hyperparams['T'][i, 0])
                for j in range(1, num_classes):
                    txt += ',' + str(hyperparams['T'][i, j])
            f.write(txt)
        
        # Saving T matrix
        T_est = torch.Tensor(T_est)
        T_err = float(np.sum(np.abs(T_est.cpu().numpy() - hyperparams['T'].cpu().numpy())) * 1.0 / num_classes * 100)
        T_errs.append(T_err)
        print('The error of T matrix is', T_err)
        if hyperparams['dataset'] == 2:
            T = hyperparams['T']
            hyperparams['T'] = T_est


        # Forward
        if hyperparams['epoch_fw'] == 0:
            # Only estimate
            with open(resultpath_times + 'Final_Results.csv', 'w') as f:
                txt = ''
                txt += 'original_ACC,' + str(originbck_top1_acc) + '\n'
                txt += 'T_error,' + str(T_err)
                f.write(txt)
        
        else:
            # Forward learning
            print('Forward learning ...')
            ep = hyperparams['epoch']
            hyperparams['epoch'] = hyperparams['epoch_fw']
            hyperparams['best_path'], ACCs_fwd, losses_fwd, best_epoch_fwd = run(train_loader, val_loader, is_forward = True, **hyperparams)
            hyperparams['epoch'] = ep
            
            # Saving results
            with open(resultpath_times + 'Forward_Train.csv', 'w') as f:
                txt = 'Epochs,Acc,Loss'
                for i in range(len(ACCs_fwd)):
                    txt += '\n' + str(i) + ',' + str(ACCs_fwd[i]) +',' + str(losses_fwd[i])
                f.write(txt)
            
            # Testing
            model = load_model(**hyperparams)
            model.load_state_dict(torch.load(hyperparams['best_path'])['state_dict'])
            model = model.to(hyperparams['device'])
            forward_top1_acc = test('T', model, test_loader, device, T_matrix = torch.eye(num_classes), is_val = False, is_fwd = True)
            ACC_fwds.append(forward_top1_acc)
            del(model)
            if hyperparams['dataset'] == 2:
                hyperparams['T'] = T

            # Saving results
            with open(resultpath_times + 'Final_Results.csv', 'w') as f:
                txt = ''
                txt += 'original_ACC,' + str(originbck_top1_acc) + '\n'
                txt += 'forward_ACC,' + str(forward_top1_acc) + '\n'
                txt += 'T_error,' + str(T_err)
                f.write(txt)

        del(train_loader)
        del(test_loader)
        del(val_loader)
    
    
    # Saving overall results
    with open(resultpath + 'Final_ACC.csv', 'w') as f:
        txt = 'methods,mean_ACC,dis_ACC\n'
        txt += 'original_back,' + str(np.array(ACC_ori).mean()) + ',' + str(np.array(ACC_ori).std()) + '\n'
        if hyperparams['epoch_fw'] != 0:
            txt += 'forward,' + str(np.array(ACC_fwds).mean()) + ',' + str(np.array(ACC_fwds).std()) + '\n'
        txt += 'T_error,' + str(np.array(T_errs).mean()) + ',' + str(np.array(T_errs).std())
        f.write(txt)