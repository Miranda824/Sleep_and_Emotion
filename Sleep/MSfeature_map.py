import matplotlib.pyplot as plt
import numpy as np
# 从文本文件中读取特征提取层的输出和预测结果
FontSize=1
feature_maps = np.loadtxt('feature_maps.txt', delimiter=',')
output = np.loadtxt('output.txt', delimiter=',')

# 绘制特征提取图
# 创建画布和子图
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
#
# # 绘制特征提取图
# ax1.plot(feature_maps)
# ax1.set_title('Feature Map')
# ax1.set_xlabel('Index')
# ax1.set_ylabel('Value')
#
# # 绘制预测结果
# ax2.plot(output)
# ax2.set_title('Output')
# ax2.set_xlabel('Index')
# ax2.set_ylabel('Value')
#
# # 调整子图之间的间距
# fig.tight_layout()
#
# # 显示图像
# plt.show()
# 绘制特征提取图
plt.figure(figsize=(10, 5))
plt.plot(feature_maps)
plt.title('Feature Map')
plt.xlabel('Index')
plt.ylabel('Value')
plt.show()

# 绘制预测结果
plt.figure(figsize=(10, 5))
plt.plot(output)
plt.title('Output')
plt.xlabel('Index')
plt.ylabel('Value')
plt.show()
