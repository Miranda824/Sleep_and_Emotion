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


# from models.Deepsleepnet import create_deepsleepnet
#
#
# def CreatNet(name):
#     if name == 'deepsleepnet':
#         input_shape = (1, 3188)  # 根据你的数据形状进行修改
#         model = create_deepsleepnet(input_shape)
#         return model
#
#     return None
#
# # 根据你的需要选择模型名称
# model_name = 'deepsleepnet'
# model = CreatNet(model_name)
