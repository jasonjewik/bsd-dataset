import time
import wandb
import torch
import logging
import torch.nn as nn
from torch.cuda.amp import autocast
from einops import rearrange

def train(epoch, model, dataloaders, optimizer, scheduler, scaler, options):
    dataloader = dataloaders["train"]
    if(options.distributed):
        dataloader.sampler.set_epoch(epoch)

    model.train()

    start = time.time()
    for index, batch in enumerate(dataloader): 
        step = dataloader.num_batches * (epoch - 1) + index
        scheduler(step)

        optimizer.zero_grad()
        
        context, target, mask = batch[0].to(options.device), batch[1].to(options.device), batch[2]["y_mask"].to(options.device)        
        target = target.nan_to_num()
        # target = torch.log(target / 86400 + 0.1) - torch.log(torch.tensor(0.1))
        # context[:, 4, :, :] = torch.log(context[:, 4, :, :] + 0.1) - torch.log(torch.tensor(0.1))

        predictions = model(context)

        with autocast():
            loss = ((torch.square(predictions - target) * (1 - mask.float())).sum([1, 2])).mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
        scaler.update()

        end = time.time()
        if(options.master and (((index + 1) % (dataloader.num_batches // 10) == 0) or (index == dataloader.num_batches - 1))):
            logging.info(f"Train epoch: {epoch:02d} [{index + 1}/{dataloader.num_batches} ({100.0 * (index + 1) / dataloader.num_batches:.0f}%)]\tLoss: {loss.item():.6f}\tTime taken {end - start:.3f}\tLearning Rate: {optimizer.param_groups[0]['lr']:.9f}")
            
            if(options.wandb):
                metrics = {"loss": loss.item(), "time": end - start, "lr": optimizer.param_groups[0]["lr"]}
                for key, value in metrics.items():
                    wandb.log({f"train/{key}": value, "step": step, "epoch": epoch})

            start = time.time()