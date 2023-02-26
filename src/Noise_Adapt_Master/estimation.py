import torch
from torch.functional import F
import numpy as np


def Anchor_est(probs):
    """
    This function is to estimate the transition matrix using anchor point
    Args:
        probs (np.array): The probabilitieso of each class
    Returns:
        trans_matrix (np.array): The estimated transition matrix.
    """
    num_classes = probs.shape[1]
    trans_matrix = np.zeros((num_classes, num_classes))
    prob_l = probs.argmax(axis = 1)

    # Estimate the transition matrix
    for i in range(num_classes):
        est_probs = probs[prob_l==i, :]
        if est_probs.shape[0] == 0:
            est_probs = probs
        anchor_index = est_probs[:, i].argmax(axis = 0)
        trans_matrix[:, i] = est_probs[anchor_index].T
    return trans_matrix


def estimator(loader, model, model_best_path, device = torch.device('cpu'), num_classes = 10):
    """
    This function caluate the flip rates.
    Args:
        loader (Dataloader): Dataloader for estimation.
        model (boject): An object of an NN model.
        model_best_path (str): The path of the best weight of the model.
        num_classes (int): The number of classes.
    Returns:
        trans_matrix (np.array): The estimated transition matrix.
    """
    model.load_state_dict(torch.load(model_best_path)["state_dict"])
    model.eval()
    probs = np.array([[0, 0, 0]])
    for step, (data, targets) in enumerate(loader):
        data = data.to(device)
        with torch.no_grad():
            outputs = model(data)

        probs = np.concatenate([probs, F.softmax(outputs, dim = 1).cpu().data.numpy()], axis = 0)
    trans_matrix = Anchor_est(probs[1:, :])
    return trans_matrix


def Anchor_est_multi(probs, rate = 0.9):
    """
    This function is to estimate the transition matrix using anchor point
    Args:
        probs (np.array): The probabilitieso of each class.
        rate (float): The threshold for selecting anchor points.
    Returns:
        trans_matrix (np.array): The estimated transition matrix.
    """
    num_classes = probs.shape[1]
    trans_matrix = np.zeros((num_classes, num_classes))
    prob_l = probs.argmax(axis = 1)

    # Estimate the transition matrix
    for i in range(num_classes):
        est_probs = probs[prob_l==i, :]
        if est_probs.shape[0] == 0:
            est_probs = probs
        est_i = est_probs[:, i]
        max_prob = est_i.max(axis = 0)
        anchor_indexes = est_i > (max_prob * 0.8)
        est_probs = est_probs[anchor_indexes]
        for j in range(est_probs.shape[0]):
            trans_matrix[:, i] += est_probs[j].T
        trans_matrix[:, i] /= est_probs.shape[0]

    return trans_matrix


def estimator_multi(loader, model, model_best_path, device = torch.device('cpu'), num_classes = 10, rate = 0.9):
    """
    This function caluate the flip rates.
    The multi-anchor estimation method is inspired by the algorithm for detection tasks (like the RCNNs) and is applied here.
    Args:
        loader (Dataloader): Dataloader for estimation.
        model (boject): An object of an NN model.
        model_best_path (str): The path of the best weight of the model.
        num_classes (int): The number of classes.
        rate (float): The threshold for selecting anchor points.
    Returns:
        trans_matrix (np.array): The estimated transition matrix.
    """
    model.load_state_dict(torch.load(model_best_path)["state_dict"])
    model.eval()
    probs = np.array([[0, 0, 0]])
    for step, (data, targets) in enumerate(loader):
        data = data.to(device)
        with torch.no_grad():
            outputs = model(data)

        probs = np.concatenate([probs, F.softmax(outputs, dim = 1).cpu().data.numpy()], axis = 0)
    trans_matrix = Anchor_est_multi(probs[1:, :], rate)
    return trans_matrix

# Demo
if __name__ == '__main__':
    c = np.array([[0, 0, 0]])
    a = np.array([[3, 2, 4],[4, 5, 6]])
    b = np.array([[21, 0, 5]])
    d = np.array([[1, 1, 9], [9.5, 0, 1], [10, 0, 2], [20, 0, 1], [19, 1, 0]])
    a = np.concatenate([a, b], axis = 0)
    c = np.concatenate([a, d], axis = 0)
    print(c)
    print(Anchor_est_multi(c))
