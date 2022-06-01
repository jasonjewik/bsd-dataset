import wandb
import torch
import logging
import torch.nn as nn
from tqdm import tqdm    
from einops import rearrange
from bsd_dataset.common.metrics import rmse, bias, pearson_correlation_coefficient

def get_metrics(model, dataloader, prefix, options):
    metrics = {}

    model.eval()

    total_rmse = 0
    total_bias = 0
    total_pearson_correlation_coefficient = 0

    with torch.no_grad():
        for batch in tqdm(dataloader):
            context, target, mask = batch[0].to(options.device), batch[1].to(options.device), batch[2]["y_mask"].to(options.device)
            target = torch.log(target / 86400 + 0.1) - torch.log(torch.tensor(0.1))
            context[:, 4, :, :] = torch.log(context[:, 4, :, :] + 0.1) - torch.log(torch.tensor(0.1))

            predictions = model(context)

            total_rmse += rmse(predictions, target) * len(target)
            total_bias += bias(predictions, target) * len(target)
            total_pearson_correlation_coefficient += pearson_correlation_coefficient(predictions, target) * len(target)

        total_rmse /= dataloader.num_samples
        total_bias /= dataloader.num_samples
        total_pearson_correlation_coefficient /= dataloader.num_samples

        metrics[f"{prefix}_rmse"] = total_rmse
        metrics[f"{prefix}_bias"] = total_bias
        metrics[f"{prefix}_pearson"] = total_pearson_correlation_coefficient

    return metrics

def evaluate(epoch, model, dataloaders, options):
    metrics = {}
    
    if(options.master):
        if(dataloaders["val"] is not None or dataloaders["test"] is not None):
            logging.info(f"Starting epoch {epoch} evaluation")

        if(dataloaders["val"] is not None): 
            metrics.update(get_metrics(model, dataloaders["val"], "val", options))
            
        if(dataloaders["test"] is not None): 
            metrics.update(get_metrics(model, dataloaders["test"], "test", options))
        
        if(metrics):
            logging.info(f"Epoch {epoch} evaluation results")
            for key, value in metrics.items():
                logging.info(f"{key}: {value:.4f}")

            if(options.wandb):
                for key, value in metrics.items():
                    wandb.log({f"evaluation/{key}": value, "epoch": epoch})

    return metrics