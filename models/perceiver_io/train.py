import argparse
from perceiver_io import *
from pos_encoding import get_fourier_position_encodings

parser = argparse.ArgumentParser()

parser.add_argument('--input_channels', type = int, default = 2, help = 'Number of input_channels')
parser.add_argument('--num_latents', type=int, default=4, help='num_latents.')
parser.add_argument('--latent_dim', type = int, default = 4, help = 'Dimension of the latent array')
parser.add_argument('--latent_heads', type = int, default = 2, help = 'Number of self attention heads')
parser.add_argument('--cross_heads', type = int, default = 2, help = 'Number of cross attention heads')
parser.add_argument('--head_dim', type = int, default = 2, help = 'Dimension of latent/cross attention head')
parser.add_argument('--depth', type = int, default = 1, help = 'Depth')
parser.add_argument('--self_per_cross_attn', type = int, default = 1, help = 'Number of self attention blocks per cross attention block')
parser.add_argument('--num_queries', type = int, default = 4, help = 'Number of query vectors')
parser.add_argument('--queries_dim', type = int, default = 258, help = 'Dimension of the query vectors')
parser.add_argument('--weight_tie_layers', action = 'store_true', default = False, help = 'allow weight sharing among different modules')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = PerceiverIO(
            dim = args.input_channels + 258, # 258 fourier position embeddings
            num_latents = args.num_latents,
            latent_dim = args.latent_dim,
            latent_heads = args.latent_heads,
            cross_heads = args.cross_heads,
            latent_dim_head = args.head_dim,
            cross_dim_head = args.head_dim,
            depth = args.depth,
            self_per_cross_attn = args.self_per_cross_attn,
            num_queries = args.num_queries,
            queries_dim = args.queries_dim,
            weight_tie_layers = args.weight_tie_layers,
        )

X = torch.rand(2, 10, 20, 2)
X_shape = X.shape
X = rearrange(X, 'b h w c -> b (h w) c')
pos_embedding = get_fourier_position_encodings(X_shape, device = device) 
X = torch.cat([X, pos_embedding], dim = -1)

print(X.shape)
print(pos_embedding.shape)
out = model(X, pos_embedding)

print(X.shape)
print(out.shape)