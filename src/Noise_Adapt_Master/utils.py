import torch
import os
import copy
import random
import numpy as np
from .algorithms_ import CNN, Resnet18, Resnet20, FCN


def fwd_loss(pred, label, T):
    """
    Calculate the forward learning loss.
    Args:
        pred (torch.tensor): Original prediction
        label (torch.tensor): Ground truth.
        T (torch.tensor): Transition matrix.
    Returns:
        loss (float): loss of the forward learning method.
    """
    print('Forward learning applying ... ')
    pred = pred @ T
    loss = torch.nn.CrossEntropyLoss()
    return loss(pred, label)


def lr_reduce(lr, epoch, total_epoch):
    """
    Learningrate reduction by linear method.
    Args:
        lr (float): Original learning rate.
        epoch (int): Current training epoch.
        total_epoch (int): The total epoch of training.
    Returns:
        lr (float): The reduced learning rate.
    """
    return lr - lr * 0.1 * epoch / total_epoch


def load_model(**hyperparams):
    """
    Load the classification model.
    Args:
        model_name (str): Name of the model.
    Returns:
        model (nn.module): A chosen classification model.
    """
    if hyperparams['model_name'] == 'resnet18':
        print('resnet18 chosen')
        return Resnet18(**hyperparams)
    elif hyperparams['model_name'] == 'resnet20':
        print('resnet20 chosen')
        return Resnet20(**hyperparams)
    elif hyperparams['model_name'] == 'FCN':
        print('FCN chosen')
        return FCN(**hyperparams)
    else:
        print('CNN chosen')
        return CNN(**hyperparams)


def get_T(**hyperparams):
    """
    Get the original T matrix.
    Args:
        name (str): Dataset name.
    Returns:
        T (torch.Tensor): T matrix.
    """
    if hyperparams['dataset'] == 0:
        T = torch.Tensor([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])
    elif hyperparams['dataset'] == 1:
        T = torch.Tensor([[0.4, 0.3, 0.3], [0.3, 0.4, 0.3], [0.3, 0.3, 0.4]])
    else:
        T = torch.eye(hyperparams['num_classes'])

    return T



def train_val_split(data, label, **hyperparams):
    """
    This is a random split function for training set and validation set.
    Args:
        data (np.array): Original dataset.
        label (np.array): Original label.
        fracs (tuple): The percentage of training set and validation set.
    Returns:
        train_data (np.array): Training set.
        train_label (np.array): Label of training set.
        val_data (np.array): Validation set.
        val_label (np.array): Label of validation set.
    """
    # Validation of parameters
    fracs = [hyperparams['train_sample'], hyperparams['val_sample']]
    assert sum(fracs) <= 1
    assert all(frac > 0 for frac in fracs)
    
    n = len(data)
    subset_len = int(n * fracs[0])
    subset_len_val = int(n * fracs[1])
    index = np.array(list(range(n)))
    np.random.shuffle(index)
    data = np.array(data)
    start_index = 0
    end_index = start_index + subset_len
    train_index = index[start_index : end_index]
    val_index = index[end_index : subset_len_val + end_index]
    train_data = data[train_index, :, :, :]
    train_label = label[train_index]
    val_data = data[val_index, :, :, :]
    val_label = label[val_index]
    return train_data, train_label, val_data, val_label


def get_device(ordinal):
    """
    This function is to identify the device of usage.
    Args:
        ordinal (int): The requested device number.
    Returns:
        device (torch.device): Device of usage.
    """
    if ordinal == '-':
        #print("Computation on CPU")
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        #print("Computation on CUDA GPU device {}".format(ordinal))
        device = torch.device('cuda:' + ordinal)
    else:
        #print("CUDA was requested but is not available! Computation will go on CPU.")
        device = torch.device('cpu')
    return device


def seed_torch(device, seed = 1029):
    """
    This function is to fix all random seeds.
    Args:
        seed (int): A fixed random seed
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device != torch.device('cpu'):
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


# Calcuate the accuracy according to the prediction and the true label.
def accuracy(output, target, topk=(1,)):
    """
    Computes the precision@k for the specified values of k.
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num):
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count