import numpy as np
import torch
import matplotlib.pyplot as plt

import transformer
from options import Options
from creatnet import CreatNet

from sklearn.manifold import TSNE

opt = Options().getparse()
# choose and creat model
opt.model_name = 'attNsoft'
net = CreatNet(opt.model_name)

if not opt.no_cuda:
    net.cuda()
if not opt.no_cudnn:
    import torch.backends.cudnn as cudnn

    cudnn.benchmark = True

# load prtrained_model
# net.load_state_dict(torch.load('./checkpoints/' + opt.model_name + '.pth'))
net.load_state_dict(torch.load('./transcheckpoint/' + 'attNsoft' + '.pth'))
net.eval()


def runmodel(eeg):
    eeg = eeg.reshape(1, -1)
    eeg = transformer.ToInputShape(eeg, opt.model_name, test_flag=True)
    eeg = transformer.ToTensor(eeg, no_cuda=opt.no_cuda)
    out = net(eeg)
    pred = torch.max(out, 1)[1]
    pred_stage = pred.data.cpu().numpy()

    out = out.data.cpu().numpy()
    # print(pred_stage[0],'---pred+_out--------------------------', out)
    return pred_stage[0], out


'''
you can change your input data here.
but the data needs meet the following conditions: 
1.fs = 100Hz
2.collect by uv
3.type   numpydata  signals:np.float16  stages:np.int16
4.shape             signals:[?,3000]   stages:[?]
'''
#载入数据集，对数据进行处理并进行了打印输出
eegdata = np.load('./datasets/simple_test/signals.npy')
true_stages = np.load('./datasets/simple_test/stages.npy')
print('shape of eegdata:', eegdata.shape)
print('shape of true_stage:', true_stages.shape)
eegdata = eegdata[:500]
true_stages = true_stages[:500]
print('shape of eegdata:', eegdata.shape)
print('shape of true_stage:', true_stages.shape)

# Normalize归一化
eegdata = transformer.Balance_individualized_differences(eegdata, '5_95_th')

# run pretrained model对每个信号进行模型预测并记录预测结果和输出结果
pred_stages = []
out_tsne = []
for i in range(len(eegdata)):
    pred_stages.append(runmodel(eegdata[i])[0])

    out_tsne.append(runmodel(eegdata[i])[1])

pred_stages = np.array(pred_stages)
#这是一个绘制嵌入图的函数
def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    dict1 = {0: 'N3', 1: 'N2', 2: 'N1', 3: 'REM', 4: 'W'}
    dict2 = {0: 'black', 1: 'darkred', 2: 'orangered', 3: 'limegreen', 4: 'fuchsia'}
    fig = plt.figure(figsize=(5, 5))
    ax = plt.subplot(111)

    for i in range(data.shape[0]):
        # plt.scatter(data[i, 0], data[i, 1], c=dict2[label[i]])
        plt.text(data[i, 0], data[i, 1], str(dict1[label[i]]),
                 color=dict2[label[i]],
                 fontdict={'weight': 'bold', 'size': 3})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig


out_tsne = np.squeeze(np.array(out_tsne), 1)
tsne = TSNE(n_components=2, init='pca', perplexity=30, random_state=0)
X = tsne.fit_transform(eegdata)
X1 = tsne.fit_transform(out_tsne)
fig = plot_embedding(X, true_stages, 'digits')

fig1 = plot_embedding(X1, true_stages, 't-SNE embedding of the digits')

plt.show()
