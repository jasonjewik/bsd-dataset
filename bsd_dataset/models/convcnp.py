import torch
import torch.nn as nn
from attrdict import AttrDict
from torch.distributions.normal import Normal
from torch.distributions.gamma import Gamma

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
    def __init__(self, num_context_features, latent_dim, kernel_size):
        super().__init__()
        self.bias = nn.Parameter(data = torch.zeros(num_context_features))        
        self.temperature = nn.Parameter(data = torch.ones(num_context_features))        
        self.conv = AbsConv(in_channels = num_context_features, out_channels = num_context_features, kernel_size = kernel_size, padding = "same", groups = num_context_features, bias = False)
        self.fc = nn.Linear(num_context_features * 2, latent_dim)

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
        self.encoder = Encoder(num_context_features = num_context_features, latent_dim = latent_dim, kernel_size = kernel_size)
        self.decoder = Decoder(latent_dim = latent_dim, kernel_size = kernel_size, num_resnet_blocks = num_resnet_blocks)
        self.ffn = FFN(input_dim = latent_dim, output_dim = output_dim, hidden_dim = hidden_dim, num_hidden_layers = num_hidden_layers)

    def forward(self, context):
        """Summary

        Args:
            context (torch.FloatTensor): batch_size(b) x num_latitudes(nx) x num_longitudes(ny) x num_context_features

        Returns:
            torch.FloatTensor: batch_size(b) x num_latitudes(nx) x num_longitudes(ny) x output_dim
        """
        context = torch.relu(self.encoder(context))
        context = torch.relu(self.decoder(context))
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
    def __init__(self):
        super().__init__()
        self.kernel = nn.ModuleList([KernelLayer() for _ in range(2)])

    def forward(self, weights, distances):
        """Summary

        Args:
            weights (torch.FloatTensor): batch_size(b) x num_latitudes(nx) x num_longitudes(ny) x 2
            distances (torch.FloatTensor): num_targets(nt) x num_latitudes(nx) x num_longitudes(ny)

        Returns:
            torch.FloatTensor: batch_size(b) x num_targets(nt) x 2
        """
        mu = self.kernel[0](weights[..., 0], distances)
        sigma = force_positive(self.kernel[1](weights[..., 1], distances))
        
        return torch.cat([mu.unsqueeze(-1), sigma.unsqueeze(-1)], dim = -1)

class GaussianConvCNP(nn.Module):
    def __init__(self, variable, num_context_features, latent_dim, kernel_size, num_resnet_blocks, hidden_dim, num_hidden_layers, num_target_features, context_lat_index, context_long_index, target_lat_index, target_long_index, history_len, temporal_context, temporal_target, **kwargs):
        super().__init__()
        self.variable = variable
        self.temporal_context = temporal_context
        self.temporal_target = temporal_target
        self.context_loc_indices = [context_lat_index, context_long_index]
        self.target_loc_indices = [target_lat_index, target_long_index]
        self.cnn = CNN(num_context_features = num_context_features * ((history_len if temporal_context else 0) + 1), latent_dim = latent_dim, kernel_size = kernel_size, num_resnet_blocks = num_resnet_blocks, output_dim = 2, hidden_dim = hidden_dim, num_hidden_layers = num_hidden_layers)
        self.gaussian = GaussianLayer()
        self.ffn = FFN(input_dim = num_target_features * ((history_len if temporal_target else 0) + 1) + 2, output_dim = 2, hidden_dim = hidden_dim, num_hidden_layers = num_hidden_layers)

    def criterion(self, outputs, labels):
        outs = AttrDict()

        batch_size = labels.shape[0]
        unsqueeze_labels = labels.reshape(-1)
        unsqueeze_outputs = outputs.reshape(-1, 2)
        distribution = Normal(loc = unsqueeze_outputs[:, 0], scale = unsqueeze_outputs[:, 1])
        log_likelihood = distribution.log_prob(unsqueeze_labels).flatten()
        outs.tar_ll = log_likelihood.sum() / batch_size 
        
        with torch.no_grad():
            if(self.variable == "tmax"):
                outs.tmax_rmse = nn.MSELoss()(outputs[..., 0], labels[..., 0])

            if(self.variable == "prep"):
                outs.precip_rmse = nn.MSELoss()(outputs[..., 0], labels[..., 0])
                
        return outs
    
    def forward(self, context, target, labels = None, predict = False):
        """Summary

        Args:
            context (torch.FloatTensor): batch_size(b) x history_len(hl) x num_time_steps(t) x num_latitudes(nx) x num_longitudes(ny) x num_context_features_per_time_step_per_history_len(ncf/t/hl)
            target  (torch.FloatTensor): batch_size(b) x history_len(hl) x num_targets(nt) x num_target_features_per_history_len(ntf/hl + 2)
            labels  (torch.FloatTensor): batch_size(b) x history_len(hl) x num_targets(nt) x 1

        Returns:
            torch.FloatTensor:
        """
        
        if(not self.temporal_context):
            context = context.unsqueeze(1)
            
        if(not self.temporal_target):
            target = target.unsqueeze(1)
            if(labels is not None): labels = labels.unsqueeze(1)
                
        context_locations = context[0, 0, 0, :, :, self.context_loc_indices]   # nx x ny x 2
        target_locations = target[0, 0, :, self.target_loc_indices]            # nt x 2
        
        context = context.permute(0, 3, 4, 5, 1, 2)                         # b x nx x ny x ncf/t/hl x hl x t
        context = context.flatten(3)                                        # b x nx x ny x ncf 
        
        weights = self.cnn(context)                                         # b x nx x ny x 2
        distances = get_distances(context_locations, target_locations)      # nt x nx x ny
        outputs = self.gaussian(weights, distances)                         # b x nt x 2

        feature_mask = torch.tensor([True] * target.shape[-1])              # ntf
        feature_mask[self.target_loc_indices] = False                       # ntf
        target_features = target[:, :, :, feature_mask]                     # b x hl x nt x ntf/hl
        target_features = target_features.permute(0, 2, 3, 1)               # b x nt x ntf/hl x hl
        target_features = target_features.flatten(2)                        # b x nt x ntf
        outputs = torch.cat([outputs, target_features], dim = -1)           # b x nt x (ntf + 2)
        outputs = self.ffn(outputs)                                         # b x nt x 2
        
        outputs[..., 1] = force_positive(outputs[..., 1])
        
        if(predict):
            return Normal(loc = outputs.unsqueeze(0)[..., :1], scale = outputs.unsqueeze(0)[..., 1:])
        
        outs = self.criterion(outputs, labels[:, -1, :, :])                               
        return outs

    def predict(self, context, target):
        with torch.no_grad():
            return self.forward(context, target, labels = None, predict = True)
    
class GammaLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel = nn.ModuleList([KernelLayer() for _ in range(3)])

    def forward(self, weights, distances):
        """Summary

        Args:
            weights (torch.FloatTensor): batch_size(b) x num_latitudes(nx) x num_longitudes(ny) x 3
            distances (torch.FloatTensor): num_targets(nt) x num_latitudes(nx) x num_longitudes(ny)

        Returns:
            torch.FloatTensor: batch_size(b) x num_targets(nt) x 3
        """
        rho = torch.sigmoid(self.kernel[0](weights[..., 0], distances))
        alpha = force_positive(self.kernel[1](weights[..., 1], distances))
        beta = force_positive(self.kernel[2](weights[..., 2], distances))

        rho = torch.clamp(rho, min = 1e-5, max = 1 - 1e-5)
        alpha = torch.clamp(alpha, min = 1e-5, max = 1e5)
        beta = torch.clamp(beta, min = 1e-5, max = 1e5)
        
        return torch.cat([rho.unsqueeze(-1), alpha.unsqueeze(-1), beta.unsqueeze(-1)], dim = -1)

class GammaConvCNP(nn.Module):
    def __init__(self, variable, num_context_features, latent_dim, kernel_size, num_resnet_blocks, hidden_dim, num_hidden_layers, num_target_features, context_lat_index, context_long_index, target_lat_index, target_long_index, history_len, temporal_context, temporal_target, **kwargs):
        super().__init__()
        self.variable = variable
        self.temporal_context = temporal_context
        self.temporal_target = temporal_target
        self.context_loc_indices = [context_lat_index, context_long_index]
        self.target_loc_indices = [target_lat_index, target_long_index]
        self.cnn = CNN(num_context_features = num_context_features * ((history_len if temporal_context else 0) + 1), latent_dim = latent_dim, kernel_size = kernel_size, num_resnet_blocks = num_resnet_blocks, output_dim = 3, hidden_dim = hidden_dim, num_hidden_layers = num_hidden_layers)
        self.gamma = GammaLayer()
        self.ffn = FFN(input_dim = num_target_features * ((history_len if temporal_target else 0) + 1) + 3, output_dim = 3, hidden_dim = hidden_dim, num_hidden_layers = num_hidden_layers)

    def criterion(self, outputs, labels):
        outs = AttrDict()
    
        batch_size = labels.shape[0]

        unsqueeze_labels = labels.reshape(-1)
        unsqueeze_outputs = outputs.reshape(-1, 3)
                
        r = torch.ones(unsqueeze_labels.shape[0]).cuda()
        r[unsqueeze_labels == 0] = 0
        unsqueeze_labels[unsqueeze_labels == 0] = 0.01

        distribution = Gamma(concentration = unsqueeze_outputs[:, 1], rate = unsqueeze_outputs[:, 2])
        log_prob = distribution.log_prob(unsqueeze_labels).flatten()
        log_likelihood = r * (nn.functional.logsigmoid(unsqueeze_outputs[:, 0]) + log_prob) + (1 - r) * nn.functional.logsigmoid(-unsqueeze_outputs[:, 0])
        
        outs.tar_ll = log_likelihood.sum() / batch_size
        
        with torch.no_grad():
            if(self.variable == "tmax"):
                outs.tmax_rmse = nn.MSELoss()(outputs[..., 1] / outputs[..., 2], labels[..., 0])

            if(self.variable == "prep"):
                outs.precip_rmse = nn.MSELoss()(outputs[..., 1] / outputs[..., 2], labels[..., 0])
                
        return outs
    
    def forward(self, context, target, labels = None, predict = False):
        """Summary

        Args:
            context (torch.FloatTensor): batch_size(b) x history_len(hl) x num_time_steps(t) x num_latitudes(nx) x num_longitudes(ny) x num_context_features_per_time_step_per_history_len(ncf/t/hl)
            target  (torch.FloatTensor): batch_size(b) x history_len(hl) x num_targets(nt) x num_target_features_per_history_len(ntf/hl + 2)
            labels  (torch.FloatTensor): batch_size(b) x history_len(hl) x num_targets(nt) x 1

        Returns:
            torch.FloatTensor:
        """
        
        if(not self.temporal_context):
            context = context.unsqueeze(1)
            
        if(not self.temporal_target):
            target = target.unsqueeze(1)
            if(labels is not None): labels = labels.unsqueeze(1)
        
        context_locations = context[0, 0, 0, :, :, self.context_loc_indices]   # nx x ny x 2
        target_locations = target[0, 0, :, self.target_loc_indices]            # nt x 2
                
        context = context.permute(0, 3, 4, 5, 1, 2)                         # b x nx x ny x ncf/t/hl x hl x t
        context = context.flatten(3)                                        # b x nx x ny x ncf 

        weights = self.cnn(context)                                         # b x nx x ny x 3
        distances = get_distances(context_locations, target_locations)      # nt x nx x ny
        outputs = self.gamma(weights, distances)                            # b x nt x 3
        
        feature_mask = torch.tensor([True] * target.shape[-1])              # ntf
        feature_mask[self.target_loc_indices] = False                       # ntf
        target_features = target[:, :, :, feature_mask]                     # b x hl x nt x ntf/hl
        target_features = target_features.permute(0, 2, 3, 1)               # b x nt x ntf/hl x hl
        target_features = target_features.flatten(2)                        # b x nt x ntf
        outputs = torch.cat([outputs, target_features], dim = -1)           # b x nt x (ntf + 3)
        outputs = self.ffn(outputs)                                         # b x nt x 3
       
        outputs[..., 0] = outputs[..., 0]
        outputs[..., 1] = force_positive(outputs[..., 1])                   
        outputs[..., 2] = force_positive(outputs[..., 2])                
        
        if(predict):
            return Gamma(concentration = outputs.unsqueeze(0)[..., 1:2], rate = outputs.unsqueeze(0)[..., 2:3])
            
        outs = self.criterion(outputs, labels[:, -1, :, :])                               
        return outs

    def predict(self, context, target):
        with torch.no_grad():
            return self.forward(context, target, labels = None, predict = True)

class CONVCNP(nn.Module):
    def __init__(self, variable, **kwargs):
        super().__init__()
        if(variable == "tmax"):
            self.convcnp = GaussianConvCNP(variable = variable, **kwargs)
        elif(variable == "prep"):
            self.convcnp = GammaConvCNP(variable = variable, **kwargs)
        else:
            raise Exception("Invalid Distribution")
    
        self.forward = self.convcnp.forward
        self.predict = self.convcnp.predict