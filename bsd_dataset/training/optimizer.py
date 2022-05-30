import torch.optim as optim

def load(model, lr, beta1, beta2, eps, weight_decay):
    parameters = ([], [])

    for name, parameter in model.named_parameters():
        if(parameter.requires_grad):
            parameters[int(any(key in name for key in ["bn", "bias"]))].append(parameter)

    optimizer = optim.AdamW([{"params": parameters[0], "weight_decay": weight_decay}, {"params": parameters[1], "weight_decay": 0}], lr = lr, betas = (beta1, beta2), eps = eps)
    return optimizer