import yaml
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from attrdict import AttrDict
import math

class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0:, :, 0::2] = torch.sin(position * div_term)
        pe[0:, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerClimate(nn.Module):
    def __init__(
        self, input_shape, target_shape, p_x, p_y,
        d_model, nhead, n_enc_layers, n_dec_layers,
        dim_ffn, dropout, activation, std_constant,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.target_shape = target_shape
        self.p_x = p_x
        self.p_y = p_y

        c = input_shape[0]
        d_in = int(p_x**2 * c)
        d_out = int(p_y**2)
        len_y = (target_shape[0]//p_y) * (target_shape[1]//p_y)

        self.d_in = d_in
        self.d_out = d_out
        self.len_y = len_y

        self.embedder = nn.Linear(d_in, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_ffn, dropout=dropout,
            activation=activation, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_enc_layers)
        self.predictor = nn.Linear(d_model, d_out) # mean and std
        self.std_constant = std_constant

        self.mask = None
    
    def get_mask(self, len_x, len_y):
        if self.mask is not None:
            return self.mask
        else:
            len = len_x + len_y
            mask = torch.zeros((len, len), device='cuda').fill_(float('-inf'))
            mask[:, 0] = 0.0
        self.mask = mask
        return self.mask

    def extract_patch(self, x, patch_size):
        x = x.cuda()
        # x: (B, H, W, C) --> (B, N, P^2 x C), where N = HW/P^2
        B, H, W, C = x.shape
        shape = [B, H // patch_size, W // patch_size] + [patch_size, patch_size] + [C]
        strides = [x.stride()[0]] + [patch_size * s for s in x.stride()[1:-1]] + list(x.stride()[1:])
        patches = torch.as_strided(x, shape, strides)
        patches = patches.reshape((B, (H//patch_size) * (W//patch_size), -1))
        return patches

    def recover_from_patch(self, x):
        b, h, w, p = x.shape[0], self.target_shape[0], self.target_shape[1], self.p_y
        shape = [b, h//p, p, w//p, p]
        stride = [x.stride()[0], w*p, p, p**2, 1]
        x = torch.as_strided(x, shape, stride=stride)
        x = x.reshape((b, h, w))
        return x

    def forward(self, x):
        x = self.extract_patch(torch.permute(x, (0, 2, 3, 1)), self.p_x)
        y = torch.zeros((x.shape[0], self.len_y, self.d_in), device=x.device)
        embeddings_x = self.embedder(x)
        embeddings_x = embeddings_x + self.pos_encoder(embeddings_x)
        embeddings_y = self.embedder(y)
        embeddings_y = embeddings_y + self.pos_encoder(embeddings_y)
        embeddings = torch.cat((embeddings_x, embeddings_y), dim=1)
        mask = self.get_mask(x.shape[1], self.len_y)

        transformer_out = self.transformer(embeddings, mask=mask)
        predictions = self.predictor(transformer_out)
        # return predictions[:, x.shape[1]:]
        return self.recover_from_patch(predictions[:, x.shape[1]:])
        # mean, std = torch.chunk(predictions, 2, dim=-1)
        # mean, std = mean[:, x.shape[1]:], std[:, x.shape[1]:]
        # if self.std_constant:
        #     std = torch.ones_like(mean)
        # else:
        #     std = torch.exp(std)
        # pred_dist = Normal(mean, std)


def Transformer(input_shape, target_shape, model_config):
    with open(model_config, "r") as f:
        config = yaml.safe_load(f)
    
    return TransformerClimate(
        input_shape=input_shape,
        target_shape=target_shape,
        p_x=config['p_x'],
        p_y=config['p_y'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        n_enc_layers=config['n_enc_layers'],
        n_dec_layers=config['n_dec_layers'],
        dim_ffn=config['dim_ffn'],
        dropout=config['dropout'],
        activation=config['activation'],
        std_constant=config['std_constant'],
    )

# from utils.misc import superseed
# superseed(0)
# model = TransformerClimate(10, 10, 20, 1, 2, 2, 20, 0.0, 'relu', True).cuda()
# x = torch.randn((4, 16, 10)).cuda()
# y_1 = torch.randn((4, 16, 10)).cuda()
# y_2 = torch.randn_like(y_1)
# _, pred_1 = model(x, y_1, torch.ones_like(y_1).bool())
# _, pred_2 = model(x, y_2, torch.ones_like(y_2).bool())
# # pred_1 = model.predict(x, y_1.shape[1])
# # pred_2 = model.predict(x, y_2.shape[1])
# # print (pred_1[0][1] == pred_2[0][1])
# print (pred_1[0][1])
# print (pred_2[0][1])