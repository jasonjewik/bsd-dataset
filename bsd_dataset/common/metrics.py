import torch
import pandas as pd

def rmse(y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.BoolTensor):
    return rmse_ignore_nans(y_pred, y_true, mask)

def bias(y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.BoolTensor):
    return bias_ignore_nans(y_pred, y_true, mask)

def pearsons_r(y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.BoolTensor):
    return pearsons_r_ignore_nans(y_pred, y_true, mask)

def mask_with_zeros(t: torch.Tensor, mask: torch.BoolTensor):
    z = torch.zeros_like(t)
    result = torch.where(mask, z, t)
    return result

def rmse_ignore_nans(y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.BoolTensor):
    mse_loss = torch.nn.MSELoss()
    masked_y_true = mask_with_zeros(y_true, mask)
    masked_y_pred = mask_with_zeros(y_pred, mask)
    return torch.sqrt(mse_loss(masked_y_pred, masked_y_true))

def bias_ignore_nans(y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.BoolTensor):
    masked_y_true = mask_with_zeros(y_true, mask)
    masked_y_pred = mask_with_zeros(y_pred, mask)
    return torch.sum(masked_y_pred - masked_y_true)

def pearsons_r_ignore_nans(y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.BoolTensor):
    masked_y_true = mask_with_zeros(y_true, mask)
    masked_y_pred = mask_with_zeros(y_pred, mask)
    y_true_df = pd.DataFrame(masked_y_true.cpu().flatten(1))
    y_pred_df = pd.DataFrame(masked_y_pred.cpu().flatten(1))    
    r = y_pred_df.corrwith(y_true_df, axis=1).mean()
    return r