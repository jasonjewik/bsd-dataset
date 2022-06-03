import wandb
import torch
import logging
import torch.nn as nn
from tqdm import tqdm    
from torchvision.transforms import GaussianBlur
from bsd_dataset.common.metrics import rmse, bias, pearsonr

def get_metrics(model, dataloader, prefix, options):
    metrics = {}

    model.eval()

    total_rmse = 0
    total_bias = 0
    total_pearsonr = 0

    with torch.no_grad():
        for batch in tqdm(dataloader):
            context, target, mask = batch[0].to(options.device), batch[1].to(options.device), batch[2]["y_mask"].to(options.device)
            target = target.nan_to_num()

            # context[:, 5, :, :] = torch.log(context[:, 5, :, :] * 86400 + 0.1) - torch.log(torch.tensor(0.1))
            # target = nn.AdaptiveAvgPool2d(output_size = target.shape[1:])(context[:, 5, :, :])
            # target = GaussianBlur(3)(nn.AdaptiveAvgPool2d(output_size = target.shape[1:])(context[:, 5, :, :]))

            if options.model == 'Transformer':
                    predictions = model.predict(context)
            else:
                predictions = model(context)

            mask[...] = False
            for i in range(len(context)):
                total_rmse += rmse(predictions[i], target[i], mask[i])
                total_bias += bias(predictions[i], target[i], mask[i])
                total_pearsonr += pearsonr(predictions[i], target[i], mask[i])

        total_rmse /= dataloader.num_samples
        total_bias /= dataloader.num_samples
        total_pearsonr /= dataloader.num_samples

        metrics[f"{prefix}_rmse"] = total_rmse
        metrics[f"{prefix}_bias"] = total_bias
        metrics[f"{prefix}_pearson_r"] = total_pearsonr

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