# import torch
# #cuda是否可用；
# torch.cuda.is_available()
#
# # 返回gpu数量；
# torch.cuda.device_count()
# # 返回gpu名字，设备索引默认从0开始；
# torch.cuda.get_device_name(0)
# # 返回当前设备索引；
# torch.cuda.current_device()
# print(torch.cuda.is_available())
# print(torch.cuda.device_count())
# print(torch.cuda.get_device_name(0))
# print(torch.cuda.get_device_name(1))
# print(torch.cuda.get_device_name(2))
# print(torch.cuda.current_device())