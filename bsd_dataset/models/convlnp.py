import math
import torch
import torch.nn as nn
from attrdict import AttrDict
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal

def force_positive(x):
    """Summary

    Args:
        x (torch.Tensor): Forces x to be positive

    Returns:
        torch.Tensor:
    """
    mask = torch.exp(x) < 1000
    x[mask] = torch.log(1 + torch.exp(x[mask]))
    return 0.9 * x + 0.01

# Not using context_locations because they are normalized (To Fix)
def get_distances(context_locations, target_locations):
    """Summary

    Args:
        context_locations (torch.FloatTensor): num_latitudes(nx) x num_longitudes(ny) x 2
        target_locations (torch.FloatTensor): num_targets(nt) x 2

    Returns:
        torch.FloatTensor: num_targets(nt) x num_latitudes(nx) x num_longitudes(ny)
    """
    # Hard coded (To Fix)
    context_lat = torch.tensor([47.109375, 48.515625, 49.921875, 51.328125, 52.734375, 54.140625]).to(target_locations.device) # nx
    context_lon = torch.tensor([ 7.03125,  8.4375 ,  9.84375, 11.25, 12.65625, 14.0625 ,15.46875]).to(target_locations.device) # ny
    
    context_locations = torch.cartesian_prod(context_lat, context_lon).view(context_lat.shape[0], context_lon.shape[0], 2) # nx x ny x 2
    
    context_locations = context_locations.unsqueeze(0)              # 1 x nx x ny x 2
    target_locations = target_locations.unsqueeze(1).unsqueeze(1)   # nt x 1 x 1 x 2
    
    context_locations = context_locations.repeat(target_locations.shape[0], 1, 1, 1)                            # nt x nx x ny x 2
    target_locations = target_locations.repeat(1, context_locations.shape[1], context_locations.shape[2], 1)    # nt x nx x ny x 2
    
    distances = (context_locations - target_locations).square().sum(-1) # nt x nx x ny
    return distances

class AbsConv(nn.Conv2d):
    def forward(self, input):
        return nn.functional.conv2d(input, self.weight.abs(), self.bias, self.stride, self.padding, self.dilation, self.groups)
    
class Encoder(nn.Module):
    def __init__(self, num_context_features, kernel_size):
        super().__init__()
        self.bias = nn.Parameter(data = torch.zeros(num_context_features))        
        self.temperature = nn.Parameter(data = torch.ones(num_context_features))        
        self.conv = AbsConv(in_channels = num_context_features, out_channels = num_context_features, kernel_size = kernel_size, padding = "same", groups = num_context_features, bias = False)
        self.fc = nn.Linear(num_context_features * 2, 2)

    def forward(self, context):
        """Summary

        Args:
            context (torch.FloatTensor): batch_size(b) x num_latitudes(nx) x num_longitudes(ny) x num_context_features(ncf)

        Returns:
            torch.FloatTensor: batch_size(b) x num_latitudes(nx) x num_longitudes(ny) x latent_dim
        """    
        self.bias.to(context.device)
        self.temperature.to(context.device)
        
        b, nx, ny, ncf = context.shape

        numerator = self.conv(context.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        denominator = self.conv(torch.ones(b, ncf, nx, ny).to(context.device)).permute(0, 2, 3, 1)

        output = numerator / torch.clamp(denominator, min = 1e-5)
        scale = torch.sigmoid((denominator.reshape(-1, ncf) + self.bias) * nn.functional.softplus(self.temperature)).view(b, nx, ny, ncf)    
        output = self.fc(torch.cat([output, scale], dim = -1))
        
        return output

class ResnetBlock(nn.Module):
    def __init__(self, latent_dim, kernel_size):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = latent_dim, out_channels = latent_dim, kernel_size = kernel_size, padding = "same", groups = latent_dim)
        self.conv2 = nn.Conv2d(in_channels = latent_dim, out_channels = latent_dim, kernel_size = 1)

    def forward(self, context):
        """Summary

        Args:
            context (torch.FloatTensor): batch_size(b) x num_latitudes(nx) x num_longitudes(ny) x latent_dim

        Returns:
            torch.FloatTensor: batch_size(b) x num_latitudes(nx) x num_longitudes(ny) x latent_dim
        """
        context = self.conv1(torch.relu(context)) + context
        context = self.conv2(context)
        return context

class Decoder(nn.Module):
    def __init__(self, latent_dim, kernel_size, num_resnet_blocks):
        super().__init__()
        self.blocks = nn.ModuleList([ResnetBlock(latent_dim, kernel_size) for _ in range(num_resnet_blocks)])

    def forward(self, context):
        """Summary

        Args:
            context (torch.FloatTensor): batch_size(b) x num_latitudes(nx) x num_longitudes(ny) x latent_dim

        Returns:
            torch.FloatTensor: batch_size(b) x num_latitudes(nx) x num_longitudes(ny) x latent_dim
        """
        context = context.permute(0, 3, 1, 2)
        for block in self.blocks:
            context = block(context)
        context = context.permute(0, 2, 3, 1)
        return context

class FFN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_hidden_layers):
        super().__init__()
        self.ih = nn.Linear(input_dim, hidden_dim)
        self.h = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden_layers)])
        self.ho = nn.Linear(hidden_dim, output_dim) 

    def forward(self, context):
        context = torch.relu(self.ih(context))
        for h in self.h: 
            context = torch.relu(h(context))
        context = self.ho(context)
        return context
    
class CNN(nn.Module):
    def __init__(self, num_context_features, latent_dim, kernel_size, num_resnet_blocks, output_dim, hidden_dim, num_hidden_layers):
        super().__init__()
        self.encoder = Encoder(num_context_features = num_context_features, kernel_size = kernel_size)
        self.fc = nn.Linear(1, latent_dim)
        self.decoder = Decoder(latent_dim = latent_dim, kernel_size = kernel_size, num_resnet_blocks = num_resnet_blocks)
        self.ffn = FFN(input_dim = latent_dim, output_dim = output_dim, hidden_dim = hidden_dim, num_hidden_layers = num_hidden_layers)

    def forward(self, context, num_samples):
        """Summary

        Args:
            context (torch.FloatTensor): batch_size(b) x num_latitudes(nx) x num_longitudes(ny) x num_context_features
            num_samples (int): 

        Returns:
            torch.FloatTensor: num_samples x batch_size(nb) x num_latitudes(nx) x num_longitudes(ny) x output_dim
        """
        x = self.encoder(context).permute(0, 3, 1, 2)
        sample = torch.normal(0, 1, (x.shape[0] * num_samples, 1, x.shape[2], x.shape[3])).to(x.device)
        mu = x[:, 0, :, :].unsqueeze(1).repeat(num_samples, 1, 1, 1)
        sigma = force_positive(x[:, 1, :, :].unsqueeze(1).repeat(num_samples, 1, 1, 1))
        sample = (sample * sigma) + mu
        z = self.fc(sample.permute(0, 2, 3, 1))
        context = torch.relu(self.decoder(z))
        context = self.ffn(context)
        return context
  
class KernelLayer(nn.Module):
    def __init__(self):        
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor([0.1]))

    def forward(self, weights, distances):
        """Summary

        Args:
            weights (torch.FloatTensor): batch_size(b) x num_latitudes(nx) x num_longitudes(ny)
            distances (torch.FloatTensor): num_targets(nt) x num_latitudes(nx) x num_longitudes(ny)

        Returns:
            torch.FloatTensor: batch_size(b) x num_targets(nt)
        """
        kernel = torch.exp(-0.5 * distances / self.scale ** 2)
        value = weights.flatten(1) @ distances.flatten(1).t()
        return value

class GaussianLayer(nn.Module):
    def __init__(self, num_variables, num_kernels):
        super().__init__()
        self.num_variables = num_variables
        self.num_kernels = num_kernels
        self.kernel = nn.ModuleList([KernelLayer() for _ in range(self.num_kernels)])

    def forward(self, weights, distances):
        """Summary

        Args:
            weights (torch.FloatTensor): batch_size(b) x num_latitudes(nx) x num_longitudes(ny) x num_variables
            distances (torch.FloatTensor): num_targets(nt) x num_latitudes(nx) x num_longitudes(ny)

        Returns:
            torch.FloatTensor: batch_size(b) x num_targets(nt) x num_variables
        """
        mus = [self.kernel[i](weights[..., i], distances).unsqueeze(-1) for i in range(self.num_variables)]
        sigmas = [force_positive(self.kernel[i](weights[..., i], distances)).unsqueeze(-1) for i in range(self.num_variables, self.num_kernels)]
        
        return torch.cat(mus + sigmas, dim = -1)

class CONVLNP(nn.Module):
    def __init__(self, num_context_features, latent_dim, kernel_size, num_resnet_blocks, hidden_dim, num_hidden_layers, num_target_features, context_lat_index, context_long_index, target_lat_index, target_long_index, history_len, temporal_context, temporal_target, variable = None, **kwargs):
        super().__init__()
        self.variable = variable
        self.temporal_context = temporal_context
        self.temporal_target = temporal_target
        self.num_variables = 1 if self.variable else 2
        self.output_dim = 2 if self.variable else 5
        self.context_loc_indices = [context_lat_index, context_long_index]
        self.target_loc_indices = [target_lat_index, target_long_index]
        self.cnn = CNN(num_context_features = num_context_features * ((history_len if temporal_context else 0) + 1), latent_dim = latent_dim, kernel_size = kernel_size, num_resnet_blocks = num_resnet_blocks, output_dim = self.output_dim, hidden_dim = hidden_dim, num_hidden_layers = num_hidden_layers)
        self.gaussian = GaussianLayer(num_variables = self.num_variables, num_kernels = self.output_dim)
        self.ffn = FFN(input_dim = self.output_dim + num_target_features * ((history_len if temporal_target else 0) + 1), output_dim = self.output_dim, hidden_dim = hidden_dim, num_hidden_layers = num_hidden_layers)        
  
    def distribution(self, outputs, num_samples):
        if(self.variable):
            outputs[..., 1] = force_positive(outputs[..., 1])
            mean, sigma = torch.chunk(outputs, 2, dim = -1)
            mean = mean.reshape(num_samples, -1, *mean.shape[1:])
            sigma = sigma.reshape(num_samples, -1, *sigma.shape[1:])
            return Normal(mean, sigma)
        else:
            mean = outputs[:, :, :2]
            sigma = outputs[:, :, 2:]
            mean = mean.reshape(num_samples, -1, *mean.shape[1:])
            sigma[:, :, 0] = force_positive(sigma[:, :, 0])
            sigma[:, :, 2] = force_positive(sigma[:, :, 2])
            sigma = torch.cat([sigma[:, :, 0].unsqueeze(-1), torch.zeros(sigma.shape[0], sigma.shape[1], 1).to(outputs.device), sigma[:, :, 1].unsqueeze(-1), sigma[:, :, 2].unsqueeze(-1)], dim = -1)
            sigma = sigma.reshape(num_samples, -1, sigma.shape[1], 2, 2)
            x = MultivariateNormal(mean, scale_tril = sigma)
            return x

    def criterion(self, outputs, labels, num_samples):
        outs = AttrDict()
        
        batch_size = labels.shape[0]
        distribution = self.distribution(outputs, num_samples)
        log_likelihood = distribution.log_prob(labels.unsqueeze(0).repeat(num_samples, 1, 1, 1))
        log_likelihood = torch.logsumexp(log_likelihood.view(num_samples, -1).sum(1), dim = 0) - math.log(num_samples)
        outs.tar_ll = log_likelihood / batch_size
        
        with torch.no_grad():
            prediction = torch.mean(distribution.loc, dim = 0)
            
            if(self.variable != "prep"):
                outs.tmax_rmse = nn.MSELoss()(prediction[..., 0], labels[..., 0])

            if(self.variable != "tmax"):
                outs.precip_rmse = nn.MSELoss()(prediction[..., -1], labels[..., -1])
                
            return outs
    
    def forward(self, context, target, labels = None, predict = False, num_samples = 64):
        """Summary

        Args:
            context (torch.FloatTensor): batch_size(b) x history_len(hl) x num_time_steps(t) x num_latitudes(nx) x num_longitudes(ny) x num_context_features_per_time_step_per_history_len(ncf/t/hl)
            target  (torch.FloatTensor): batch_size(b) x history_len(hl) x num_targets(nt) x num_target_features_per_history_len(ntf/hl + 2)
            labels  (torch.FloatTensor): batch_size(b) x history_len(hl) x num_targets(nt) x 1
            num_samples (int)
        Returns:
            torch.FloatTensor:
        """
        
        if(not self.temporal_context):
            context = context.unsqueeze(1)
            
        if(not self.temporal_target):
            target = target.unsqueeze(1)
            if(labels is not None): labels = labels.unsqueeze(1)
            
        context_locations = context[0, 0, 0, :, :, self.context_loc_indices]
        target_locations = target[0, 0, :, self.target_loc_indices]
                
        context = context.permute(0, 3, 4, 5, 1, 2)
        context = context.flatten(3)
        
        weights = self.cnn(context, num_samples = num_samples)
        distances = get_distances(context_locations, target_locations)
        outputs = self.gaussian(weights, distances)

        feature_mask = torch.tensor([True] * target.shape[-1])
        feature_mask[self.target_loc_indices] = False
        target_features = target[:, :, :, feature_mask]
        target_features = target_features.permute(0, 2, 3, 1)
        target_features = target_features.flatten(2)
        outputs = torch.cat([outputs, target_features.repeat(num_samples, 1, 1)], dim = -1)
        outputs = self.ffn(outputs)
                
        if(predict):
            return self.distribution(outputs, num_samples)
        
        outs = self.criterion(outputs, labels[:, -1, :, :], num_samples)                               
        return outs

    def predict(self, context, target, num_samples = 64):
        with torch.no_grad():
            return self.forward(context, target, labels = None, predict = True, num_samples = num_samples)