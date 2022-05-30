from ..models import *

def load(model, **kwargs):
    if(model == "ConvNet1"):
        return ConvNet1(input_shape = kwargs["input_shape"], output_shape = kwargs["output_shape"])
    if(model == "ConvNet2"):
        return ConvNet2(input_shape = kwargs["input_shape"], output_shape = kwargs["output_shape"])
    if(model == "ConvNet3"):
        return ConvNet3(input_shape = kwargs["input_shape"], output_shape = kwargs["output_shape"])