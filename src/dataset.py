import numpy as np
from torch.utils.data import Dataset, DataLoader

from Noise_Adapt_Master.utils import train_val_split


class Label_Noise_Dataset(Dataset):
    """
    This is a child class of torch.Dataset. This class is for datasets
    Args:
        data (np.array): Origiinal dta.
        gt (np.array): Original label sequence.
        dataset (str): name of dataset.
        flip_augmentation (bool): Control parameter of whether or not doing random flip of the data.
    """
    def __init__(self, data, gt, **hyperparams):
        self.data = data
        self.label = gt
        self.flip_augmentation = hyperparams['flip_augmentation']
        print(hyperparams['name'], 'dataset is loaded with the shape of', self.data.shape, '\nDo a random flip of data\n' if self.flip_augmentation else '\n')
        
        self.indices = np.array(range(len(self.label)))
        np.random.shuffle(self.indices)


    # Random flip with the probability of p
    @staticmethod
    def flip(arrays, p = 0.3):
        horizontal = np.random.random() > p
        vertical = np.random.random() > p
        if horizontal:
            arrays = [np.fliplr(arr) for arr in arrays]
        if vertical:
            arrays = [np.flipud(arr) for arr in arrays]
        return np.array(arrays)


    def rescale(self, arrays):
        min_ = arrays.min()
        max_ = arrays.max()
        arrays -= min_
        arrays = arrays / (max_ - min_)
        return arrays


    def __len__(self):
        return len(self.label.copy())
    
    
    def __getitem__(self, index):
        data_ = self.data.copy()
        label_ = self.label.copy()
        data = data_[self.indices[index]]
        label = label_[self.indices[index]].astype(np.int64)
        if self.flip_augmentation:
            data = self.flip(data)
        data = self.rescale(data).astype(np.float32)
        return (data, label)


def make_loader(data, label, split, **hyperparams):
    """
    This function makes the dataset into torch Dataloader.
    If it is the training data, this function will split the dataset into training set and validation set.
    Args:
        data (np.array): Original dataset.
        label (np.array): Original labels.
        split (bool): Control parameter of whether the dataset needs to be split. Set False when making test loader.
        hyperparams (dict): Hyperparameters of the dataset.
    Returns:
        loader_list (tuple): The tuple of Dataloader.
    """
    loader_list = []
    data = data.reshape(data.shape[0], -1, data.shape[1], data.shape[2])
    num_workers = hyperparams['num_workers']

    if split:
        train_data, train_label, val_data, val_label = train_val_split(data, label, **hyperparams)
        train_data = Label_Noise_Dataset(train_data, train_label, **hyperparams)
        val_data = Label_Noise_Dataset(val_data, val_label, **hyperparams)
        
        loader_list.append(DataLoader(
            train_data,
            batch_size = hyperparams['batch_size'],
            shuffle = True,
            num_workers = num_workers,
            pin_memory = True
        ))

        loader_list.append(DataLoader(
            val_data,
            batch_size = 5000,
            shuffle = False,
            num_workers = num_workers,
            pin_memory = True
        ))

    else:
        test_data = Label_Noise_Dataset(data, label, **hyperparams)
        loader_list.append(DataLoader(
            test_data,
            batch_size = 5000,
            num_workers = num_workers,
            shuffle = False,
            pin_memory = True
        ))

    return loader_list

