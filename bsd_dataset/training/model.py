from ..models import *

def load(model, **kwargs):
    if(model == "ConvNet"):
        return ConvNet(input_shape = kwargs["input_shape"], target_shape = kwargs["target_shape"])
    if(model == "GaussianConvNet"):
        return GaussianConvNet(input_shape = kwargs["input_shape"], target_shape = kwargs["target_shape"])
    if(model == "PerceiverIO"):
        return Perceiver(input_shape = kwargs["input_shape"], target_shape = kwargs["target_shape"], model_config = kwargs["model_config"])
    raise Exception(f"Model {model} is not supported")