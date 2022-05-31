import yaml

dict_file = [{'name': ['PERCEIVER_IO']}, {'details': [{'num_latents': 4, 'latent_dim': 64, 
                'latent_heads': 2, 'cross_heads': 2, 'head_dim': 32, 'depth':6,
                'self_per_cross_attn': 2, 'weight_tie_layers': False, 'pos_encoding': 64}]}]

with open('perceiver_io.yml', 'w') as f:
    documents = yaml.dump(dict_file, f)
# with open('perceiver_io.yml', 'r') as f:
#         dict_file = yaml.full_load(f)
#         configs = dict_file[1]['details'][0]
#         print(configs)
