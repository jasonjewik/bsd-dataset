import torch
import numpy as np
import pandas as pd

def nan_to_num(t, mask=None):
    if mask is None:
        mask = torch.isnan(t)
    zeros = torch.zeros_like(t)
    return torch.where(mask, zeros, t)

def rmse(y_pred, y_true):
    mse_loss = torch.nn.MSELoss()
    if type(y_pred) == np.ndarray:
        y_pred = torch.tensor(y_pred)
    if type(y_true) == np.ndarray:
        y_true = torch.tensor(y_true)
    y_pred = nan_to_num(y_pred, torch.isnan(y_true))
    y_true = nan_to_num(y_true)
    return torch.sqrt(mse_loss(y_pred, y_true))

def bias(y_pred, y_true):
    if type(y_pred) == np.ndarray:
        y_pred = torch.tensor(y_pred)
    if type(y_true) == np.ndarray:
        y_true = torch.tensor(y_true)
    y_pred = nan_to_num(y_pred, torch.isnan(y_true))
    y_true = nan_to_num(y_true)
    return torch.sum(y_pred - y_true)

def pearson_correlation_coefficient(y_pred, y_true):
    if type(y_pred) == np.ndarray:
        y_pred = torch.tensor(y_pred)
    if type(y_true) == np.ndarray:
        y_true = torch.tensor(y_true)
    y_pred = nan_to_num(y_pred, torch.isnan(y_true))
    y_true = nan_to_num(y_true)
    y_pred = pd.DataFrame(y_pred.cpu().flatten(1))
    y_true = pd.DataFrame(y_true.cpu().flatten(1))
    r = y_pred.corrwith(y_true, axis=1).mean()
    return r