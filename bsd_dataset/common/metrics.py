import torch

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