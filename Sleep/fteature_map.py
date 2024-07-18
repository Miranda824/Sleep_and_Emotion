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
net.load_state_dict(torch.load('./checkpoints/' + opt.model_name + '.pth'))
net.eval()

# The input data eeg is a 1-dimensional vector, which is reshaped into a matrix of 1 row and n columns, where n is the data length.
# Use the transformer.ToInputShape function to preprocess eeg and convert it to the shape required for the input model. The specific operation and implementation are the same as the ToInputShape function, which will not be repeated here.
# Use the transformer.ToTensor function to convert the preprocessed data into a tensor type, and you can choose whether to put it into the GPU for calculation.
# Input the tensor type data into the deep learning model to get the model output out.
# Perform the argmax operation on the model output to get the prediction result pred.
# Return the prediction result and model output in the form of a tuple. Among them, the prediction result pred is an integer, which represents the predicted classification result of the eeg data; 
# the model output out is a numpy matrix with 1 row and multiple columns, which represents the probability that the eeg data is predicted as each classification.
def runmodel(eeg):
    eeg = eeg.reshape(1, -1)
    eeg = transformer.ToInputShape(eeg, opt.model_name, test_flag=True)
    eeg = transformer.ToTensor(eeg, no_cuda=opt.no_cuda)
    out = net(eeg)
    pred = torch.max(out, 1)[1]
    pred_stage = pred.data.cpu().numpy()

    out = out.data.cpu().numpy()

    return pred_stage[0], out


'''
you can change your input data here.
but the data needs meet the following conditions: 
1.fs = 100Hz
2.collect by uv
3.type   numpydata  signals:np.float16  stages:np.int16
4.shape             signals:[?,3000]   stages:[?]
'''
# Load the data set, process the data and print it out
eegdata = np.load('./datasets/simple_test/signals.npy')
true_stages = np.load('./datasets/simple_test/stages.npy')
print('shape of eegdata:', eegdata.shape)
print('shape of true_stage:', true_stages.shape)
eegdata = eegdata[:250]
true_stages = true_stages[:250]
print('shape of eegdata:', eegdata.shape)
print('shape of true_stage:', true_stages.shape)

# Normalize
eegdata = transformer.Balance_individualized_differences(eegdata, '5_95_th')

# run pretrained model
pred_stages = []
out_tsne = []
for i in range(len(eegdata)):
    pred_stages.append(runmodel(eegdata[i])[0])

    out_tsne.append(runmodel(eegdata[i])[1])

pred_stages = np.array(pred_stages)

# Plotting the embedding graph
def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    dict1 = {0: 'N3', 1: 'N2', 2: 'N1', 3: 'REM', 4: 'W'}
    dict2 = {0: 'darkviolet', 1: 'lime', 2: 'indigo', 3: 'olive', 4: 'crimson'}
    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        # plt.scatter(data[i, 0], data[i, 1], c=dict2[label[i]])
        plt.text(data[i, 0], data[i, 1], str(dict1[label[i]]),
                 color=dict2[label[i]],
                 fontdict={'weight': 'bold', 'size': 9})
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

print('err:', sum((true_stages[i] != pred_stages[i]) for i in range(len(pred_stages))) / len(true_stages) * 100, '%')
