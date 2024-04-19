import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

# 载入预测结果和模型输出结果
stage = np.loadtxt('stage.txt', dtype=np.int)
out = np.loadtxt('out.txt', dtype=np.float)
eeg =np.loadtxt('signal.txt', dtype=np.float)
eeg = reshape(1,-1)
pred = torch.max(out, 1)[1]
pred_stage = pred.data.cpu().numpy()

# 载入真实结果
true_stages = np.loadtxt('stage.txt', dtype=np.int)
true_stages = true_stages[:250]

# 这是一个绘制嵌入图的函数
def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    dict1 = {0: 'N3', 1: 'N2', 2: 'N1', 3: 'REM', 4: 'W'}
    dict2 = {0: 'darkviolet', 1: 'lime', 2: 'indigo', 3: 'olive', 4: 'crimson'}
    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(dict1[label[i]]),
                 color=dict2[label[i]],
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig

# 使用 t-SNE 将原始数据和输出结果进行降维
tsne = TSNE(n_components=2, init='pca', perplexity=30, random_state=0)
X = tsne.fit_transform(eegdata)
X1 = tsne.fit_transform(out_tsne)

# 绘制嵌入图
fig = plot_embedding(X, true_stages, 'digits')
fig1 = plot_embedding(X1, true_stages, 't-SNE embedding of the digits')
plt.show()

# 计算预测结果和真实结果之间的错误率
error_rate = np.sum(true_stages != pred_stages) / len(true_stages) * 100
print('err:', error_rate, '%')

# 绘制真实结果和预测结果的睡眠分期曲线图
plt.figure()
plt.xlim((0, len(true_stages)))
plt.ylim((0, 6))
plt.plot(true_stages + 1, color='blue', label="Expert")
plt.plot(pred_stages + 1, color='orange', label="Our model")
plt.legend(loc="lower right")
plt.yticks([1, 2, 3, 4, 5], ['N3', 'N2', 'N1', 'REM', 'W'])
plt.xlabel('Epoch number')
plt.show()

