from torch import nn
from models import attNsoft1
# from models import attNsoft1


from models import attNsoft1

def CreatNet(name):
    if name == 'attNsoft1':
        net = attNsoft1.Multi_Scale_ResNet(inchannel=1, num_classes=5)
    else:
        raise ValueError(f"Invalid network name: {name}")

    return net
