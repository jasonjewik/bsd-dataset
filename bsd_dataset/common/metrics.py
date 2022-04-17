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
    y_pred = torch.tensor(y_pred)
    y_true = torch.tensor(y_true)
    y_pred = nan_to_num(y_pred, torch.isnan(y_true))
    y_true = nan_to_num(y_true)
    return torch.sqrt(mse_loss(y_pred, y_true))

def bias(y_pred, y_true):
    return np.sum(y_pred - y_true)

def pearson_correlation_coefficient(y_pred, y_true):
    y_pred = pd.Dataframe(y_pred)
    y_true = pd.Dataframe(y_true)
    r = y_pred.corrwith(y_true, axis=1).mean()
    return r