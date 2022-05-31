import numpy as np
import torch
from einops import rearrange, repeat

ff_input = None
ff_target = None

def get_fourier_position_encodings(args, num_bands = 64, device = 'cpu', input = True):

    global ff_input, ff_target
    ff = ff_input if input else ff_target
    b, d, h, w = args
    if ff is None:    
        pos = build_linear_positions((h, w))
        pos = rearrange(pos, 'h w d -> (h w) d')
        ff = generate_fourier_features(pos, num_bands) # use 64 bands for position embedding
    encoding = repeat(ff, 'n d -> b n d', b = b)
    encoding = torch.from_numpy(encoding).to(device).float()    
    return encoding
    
def build_linear_positions(index_dims, output_range=(-1.0, 1.0)):
  """Generate an array of position indices for an N-D input array.
  Args:
    index_dims: The shape of the index dimensions of the input array.
    output_range: The min and max values taken by each input index dimension.
  Returns:
    A np array of shape [index_dims[0], index_dims[1], .., index_dims[-1], N].
  """
  def _linspace(n_xels_per_dim):
    return np.linspace(
        output_range[0], output_range[1],
        num=n_xels_per_dim,
        endpoint=True, dtype=np.float32)

  dim_ranges = [
      _linspace(n_xels_per_dim) for n_xels_per_dim in index_dims]
  array_index_grid = np.meshgrid(*dim_ranges, indexing='ij')

  return np.stack(array_index_grid, axis=-1)


def generate_fourier_features(
    pos, num_bands, max_resolution=(224, 224),
    concat_pos=True, sine_only=False):
  """Generate a Fourier frequency position encoding with linear spacing.
  Args:
    pos: The position of n points in d dimensional space.
      A jnp array of shape [n, d].
    num_bands: The number of bands (K) to use.
    max_resolution: The maximum resolution (i.e. the number of pixels per dim).
      A tuple representing resolution for each dimension
    concat_pos: Concatenate the input position encoding to the Fourier features?
    sine_only: Whether to use a single phase (sin) or two (sin/cos) for each
      frequency band.
  Returns:
    embedding: A 1D jnp array of shape [n, n_channels]. If concat_pos is True
      and sine_only is False, output dimensions are ordered as:
        [dim_1, dim_2, ..., dim_d,
         sin(pi*f_1*dim_1), ..., sin(pi*f_K*dim_1), ...,
         sin(pi*f_1*dim_d), ..., sin(pi*f_K*dim_d),
         cos(pi*f_1*dim_1), ..., cos(pi*f_K*dim_1), ...,
         cos(pi*f_1*dim_d), ..., cos(pi*f_K*dim_d)],
       where dim_i is pos[:, i] and f_k is the kth frequency band.
  """
  min_freq = 1.0
  # Nyquist frequency at the target resolution:

  freq_bands = np.stack([
      np.linspace(min_freq, res / 2, num=num_bands, endpoint=True)
      for res in max_resolution], axis=0)

  # Get frequency bands for each spatial dimension.
  # Output is size [n, d * num_bands]
  per_pos_features = pos[:, :, None] * freq_bands[None, :, :]
  per_pos_features = np.reshape(per_pos_features,
                                 [-1, np.prod(per_pos_features.shape[1:])])

  if sine_only:
    # Output is size [n, d * num_bands]
    per_pos_features = np.sin(np.pi * (per_pos_features))
  else:
    # Output is size [n, 2 * d * num_bands]
    per_pos_features = np.concatenate(
        [np.sin(np.pi * per_pos_features),
         np.cos(np.pi * per_pos_features)], axis=-1)
  # Concatenate the raw input positions.
  if concat_pos:
    # Adds d bands to the encoding.
    per_pos_features = np.concatenate([pos, per_pos_features], axis=-1)
  return per_pos_features
