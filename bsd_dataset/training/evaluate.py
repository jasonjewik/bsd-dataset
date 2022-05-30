import wandb
import torch
import logging
import torch.nn as nn
from tqdm import tqdm    

def get_val_metrics(model, dataloader, options):
    logging.info("Started validation")

    metrics = {}

    model.eval()
    criterion = nn.MSELoss(reduction = "none").to(options.device)

    losses = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            context, target, mask = batch[0].to(options.device), batch[1].to(options.device), batch[2]["y_mask"].to(options.device)
            predictions = model(context)
            loss = (criterion(predictions, target).nan_to_num() * (1 - mask.float())).sum([1, 2]) / (1 - mask.float()).sum([1, 2])
            losses.append(loss.sum())

        loss = sum(losses) / dataloader.num_samples
        metrics["val_loss"] = loss

    logging.info("Finished validation")

    return metrics

def get_test_metrics(model, dataloader, options):
    logging.info("Started testing")

    metrics = {}

    model.eval()
    criterion = nn.MSELoss(reduction = "none").to(options.device)

    losses = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            context, target, mask = batch[0].to(options.device), batch[1].to(options.device), batch[2]["y_mask"].to(options.device)
            predictions = model(context)
            loss = (criterion(predictions, target).nan_to_num() * (1 - mask.float())).sum([1, 2]) / (1 - mask.float()).sum([1, 2])
            losses.append(loss.sum())

        loss = sum(losses) / dataloader.num_samples
        metrics["test_loss"] = loss
    
    logging.info("Finished testing")

    return metrics

def evaluate(epoch, model, dataloaders, options):
    metrics = {}
    
    if(options.master):
        if(dataloaders["val"] is not None or dataloaders["test"] is not None):
            if(epoch == 0):
                logging.info(f"Base evaluation")
            else:
                logging.info(f"Epoch {epoch} evaluation")

        if(dataloaders["val"] is not None): 
            metrics.update(get_val_metrics(model, dataloaders["val"], options))
            
        if(dataloaders["test"] is not None): 
            metrics.update(get_test_metrics(model, dataloaders["test"], options))
        
        if(metrics):
            logging.info("Evaluation")
            for key, value in metrics.items():
                logging.info(f"{key}: {value:.4f}")

            if(options.wandb):
                for key, value in metrics.items():
                    wandb.log({f"evaluation/{key}": value, "epoch": epoch})

    return metrics