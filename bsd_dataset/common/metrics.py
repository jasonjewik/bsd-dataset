import pandas as pd
from scipy import stats
import torch

def rmse(y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.BoolTensor):
    return rmse_ignore_nans(y_pred, y_true, mask)

def bias(y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.BoolTensor):
    return bias_ignore_nans(y_pred, y_true, mask)

def pearsonr(y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.BoolTensor):
    return pearsonr_ignore_nans(y_pred, y_true, mask)

def mask_with_zeros(t: torch.Tensor, mask: torch.BoolTensor):
    z = torch.zeros_like(t)
    result = torch.where(mask, z, t)
    return result

def rmse_ignore_nans(y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.BoolTensor):
    masked_y_true = y_true[~mask]
    masked_y_pred = y_pred[~mask]
    return (masked_y_pred - masked_y_true).square().mean().sqrt().cpu().item()

def bias_ignore_nans(y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.BoolTensor):
    masked_y_true = y_true[~mask]
    masked_y_pred = y_pred[~mask]
    return (masked_y_pred - masked_y_true).abs().mean().cpu().item()

def pearsonr_ignore_nans(y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.BoolTensor):
    masked_y_true = y_true[~mask]
    masked_y_pred = y_pred[~mask]
    return stats.pearsonr(masked_y_true.cpu().numpy(), masked_y_pred.cpu().numpy())[0]