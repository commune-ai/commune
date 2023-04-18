from accelerate import init_empty_weights
from torch import nn
import commune

with init_empty_weights():
    model = commune.get_module('model.transformer.llama')()
    
print(commune.get_model_size(model))
    
