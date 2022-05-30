from ..models import *

def load(model, **kwargs):
    if(model == "ConvNet"):
        return ConvNet(input_shape = kwargs["input_shape"], target_shape = kwargs["target_shape"])
    if(model == "GaussianConvNet"):
        return GaussianConvNet(input_shape = kwargs["input_shape"], target_shape = kwargs["target_shape"])
    if(model == "GammaConvNet"):
        return GammaConvNet(input_shape = kwargs["input_shape"], target_shape = kwargs["target_shape"])
    raise Exception(f"Model {model} is not supported")