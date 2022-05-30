import wandb
import torch
import logging
import torch.nn as nn
from tqdm import tqdm    

def get_validation_metrics(model, dataloader, options):
    logging.info("Started validating")

    metrics = {}

    model.eval()
    criterion = nn.MSELoss(reduction = "sum").to(options.device)

    losses = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            context, target, mask = batch[0].to(options.device), batch[1].to(options.device), batch[2].to(options.device)
            predictions = model(context)
            loss = (criterion(predictions, target) * mask.float()).sum() / mask.sum()
            losses.append(loss)

        loss = sum(losses) / dataloader.num_samples
        metrics["validation_loss"] = loss

    logging.info("Finished validating")

    return metrics

def get_test_metrics(model, dataloader, options):
    logging.info("Started testing")

    metrics = {}

    model.eval()
    criterion = nn.MSELoss(reduction = "sum").to(options.device)

    losses = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            context, target, mask = batch[0].to(options.device), batch[1].to(options.device), batch[2].to(options.device)
            predictions = model(context)
            loss = (criterion(predictions, target) * mask.float()).sum() / mask.sum()
            losses.append(loss)

        loss = sum(losses) / dataloader.num_samples
        metrics["test_loss"] = loss
    
    logging.info("Finished testing")

    return metrics

def evaluate(epoch, model, data, options):
    metrics = {}
    
    if(options.master):
        if(data["validation"] is not None or data["test"] is not None):
            if(epoch == 0):
                logging.info(f"Base evaluation")
            else:
                logging.info(f"Epoch {epoch} evaluation")

        if(data["validation"] is not None): 
            metrics.update(get_validation_metrics(model, data["validation"], options))
            
        if(data["test"] is not None): 
            metrics.update(get_test_metrics(model, data["eval_test"], options))
        
        if(metrics):
            logging.info("Evaluation")
            for key, value in metrics.items():
                logging.info(f"{key}: {value:.4f}")

            if(options.wandb):
                for key, value in metrics.items():
                    wandb.log({f"evaluation/{key}": value, "epoch": epoch})

    return metrics