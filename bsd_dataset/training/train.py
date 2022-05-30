import time
import wandb
import torch
import logging
import torch.nn as nn
from torch.cuda.amp import autocast

def train(epoch, model, dataloader, optimizer, scheduler, scaler, options):    
    if(options.distributed):
        dataloader.sampler.set_epoch(epoch)

    model.train()
    criterion = nn.MSELoss(reduction = "none").to(options.device)

    start = time.time()
    for index, batch in enumerate(dataloader): 
        step = len(dataloader) * epoch + index
        scheduler(step)

        optimizer.zero_grad()
        
        context, target, mask = batch[0].to(options.device), batch[1].to(options.device), batch[2].to(options.device)
        predictions = model(context)

        with autocast():
            loss = (criterion(predictions, target) * mask.float()).sum() / mask.sum()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
        
        scaler.update()

        end = time.time()
        if(options.master and (((index + 1) % (len(dataloader) // 10) == 0) or (index == len(dataloader) - 1))):
            logging.info(f"Train epoch: {epoch:02d} [{index + 1}/{len(dataloader)} ({100.0 * (index + 1) / len(dataloader):.0f}%)]\tLoss: {loss.item():.6f}\tTime taken {end - start:.3f}\tLearning Rate: {optimizer.param_groups[0]['lr']:.9f}")
            
            if(options.wandb):
                metrics = {"loss": loss.item(), "time": end - start, "lr": optimizer.param_groups[0]["lr"]}
                for key, value in metrics.items():
                    wandb.log({f"train/{key}": value, "step": step, "epoch": epoch})

            start = time.time()