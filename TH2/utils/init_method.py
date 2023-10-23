import torch.nn as nn
def ones_init(model):
    nn.init.ones_(model.weight)
    nn.init.ones_(model.bias)
    return model

def xavier_init(model):
    nn.init.xavier_uniform_(model.weight)
    nn.init.zeros_(model.bias)
    return model