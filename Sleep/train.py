import time
import numpy as np
import torch
from torch import nn
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")
import os
import util
import transformer
import dataloader
import statistics
import heatmap
from creatnet import CreatNet
from options import Options
from rocNpr import roc_plot, pr_plot
from tensorflow.keras.models import load_model

if hasattr(torch.cuda, 'empty_cache'):
    torch.cuda.empty_cache()
# Set visible GPU devices
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"  # Set the visible GPU device number
emotion_model = load_model('F:/sleep/Sleep/emo_model/emotion_re_15surprise.h5', compile=False)



opt = Options().getparse()
localtime = time.asctime(time.localtime(time.time()))
util.writelog('\n\n' + str(localtime) + '\n' + str(opt))
t1 = time.time()

'''
change your own data to train
but the data needs meet the following conditions: 
1.type   numpydata  signals:np.float16  stages:np.int16
2.shape             signals:[?,3000]   stages:[?]
3.fs = 100Hz
4.input signal data should be normalized!!
  we recommend signal data normalized useing 5_95_th for each subject, 
  example: signals_subject=Balance_individualized_differences(signals_subject, '5_95_th')
5.when useing subject cross validation,we will generally believe [0:80%]and[80%:100%]data come from different subjects.
'''
signals, stages = dataloader.loaddataset(opt.dataset_dir, opt.dataset_name, opt.signal_name, opt.sample_num, opt.BID,
                                         opt.select_sleep_time, shuffle=True,model=emotion_model)
stage_cnt, stage_cnt_per = statistics.stage(stages)

# np.save("./datasets/simple_test/signals.npy", np.array(list(signals)))
# np.save("./datasets/simple_test/stages.npy", np.array(list(stages)))

if opt.cross_validation == 'k_fold':
    signals, stages = transformer.batch_generator(signals, stages, opt.batchsize, shuffle=True)
    print('------------------here-------------------:',signals.shape)
    train_sequences, test_sequences = transformer.k_fold_generator(len(stages), opt.fold_num)
elif opt.cross_validation == 'subject':
    util.writelog('train statistics:', True)
    stage_cnt, stage_cnt_per = statistics.stage(stages[:int(0.95 * len(stages))])
    util.writelog('test statistics:', True)
    stage_cnt, stage_cnt_per = statistics.stage(stages[int(0.95 * len(stages)):])
    signals, stages = transformer.batch_generator_subject(signals, stages, opt.batchsize, shuffle=False)
    train_sequences, test_sequences = transformer.k_fold_generator(len(stages), 1)

batch_length = len(stages)
print('length of batch:', batch_length)
show_freq = int(len(train_sequences[0]) / 5)
t2 = time.time()
print('load data cost time: %.2f' % (t2 - t1), 's')

net = CreatNet(opt.model_name)
torch.save(net.cpu().state_dict(), './checkpoints/' + opt.model_name + '.pth')
util.show_paramsnumber(net)

weight = np.array([1, 1, 1, 1, 1])
if opt.weight_mod == 'avg_best':
    weight = np.log(1 / stage_cnt_per)
    # weight[2] = weight[2] + 1
    weight = np.clip(weight, 1, 5)
print('Loss_weight:', weight)
weight = torch.from_numpy(weight).float()

# print(net)

if not opt.no_cuda:
    net.cuda()
    weight = weight.cuda()
if not opt.no_cudnn:
    torch.backends.cudnn.benchmark = True

#optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr)
optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr)


# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
criterion = nn.CrossEntropyLoss(weight)

# ——_——_——_——_——_——_——_——————_——_——Encapsulating Models with DataParallel_——__——_
# model.train()  # Change net to model
#
# net = net.to(device)
# -----------------------------------
def evalnet(net, signals, stages, sequences, epoch, plot, plot_result={}, save_file=None):
    net.eval()
    confusion_mat = torch.zeros((5,5),dtype=int).to('cuda:0')

    for i, sequence in enumerate(sequences, 1):

        signal = transformer.ToInputShape(signals[sequence], opt.model_name, test_flag=True)
        signal, stage = transformer.ToTensor(signal, stages[sequence], no_cuda=opt.no_cuda)
        # signal = signal.to(device)  # Move input data to the GPU device
        # stage = stage.to(device)  # Move label data to GPU device
        with torch.no_grad():
            out = net(signal)
            # out=model(signal)

        pred = torch.max(out, 1)[1]
        pred = pred.data.cpu().numpy()
        stage = stage.data.cpu().numpy()
        out = out.data.cpu().numpy()

        roc_plot(out, stage)
        pr_plot(out, stage)

        for x in range(len(pred)):
            confusion_mat[stage[x]][pred[x]] += 1

    recall, acc, sp, err, k = statistics.result(confusion_mat)
    plot_result['test'].append(err)
    heatmap.draw(confusion_mat, name='test')
    print('recall,acc,sp,err,k: ' + str(statistics.result(confusion_mat)))
    return plot_result, confusion_mat


# Call the evalnet function with the save_file parameter
# confusion_mat = evalnet(net, signals, stages,epoch,plot save_file)


print('begin to train ...')
# First, initialize a 5*5 two-dimensional array final_confusion_mat and assign all its elements to 0, indicating the initial value of the 5-class confusion matrix to be calculated later.
# Then opt.fold_num loops are performed, each of which loads the pre-trained model parameters and sets the model based on whether the GPU is used.
# Next, an empty list confusion_mats is defined to store the confusion matrix results for each fold.

final_confusion_mat = np.zeros((5, 5), dtype=int)


# Create the tqdm object before the loop
epochs_range = tqdm(range(opt.epochs), desc="Epochs", unit="epoch")
for fold in range(opt.fold_num):
    # net.load_state_dict(torch.load('./checkpoints/' + opt.model_name + '.pth'))
    # if opt.pretrained:
    #     net.load_state_dict(torch.load('./checkpoints/pretrained/' + opt.dataset_name + '/' + opt.model_name + '.pth'))
    if not opt.no_cuda:
        net.cuda()
    plot_result = {'train': [1.], 'test': [1.]}
    confusion_mats = []
    plot = 'false'  # Save data in the last epoch to draw multiple roc, pr

    # Create a tqdm object for the current fold
    confusion_mat = torch.zeros((5,5),dtype=int).to('cuda:0')
    epochs_range = tqdm(range(opt.epochs), desc=f"Fold {fold + 1} Epochs", unit="epoch")
    for epoch in epochs_range:
        t1 = time.time()
        confusion_mat = np.zeros((5, 5), dtype=int)
        print('fold:', fold + 1, 'epoch:', epoch + 1)
        net.train()
        for i, sequence in enumerate(train_sequences[fold], 1):
            signal = transformer.ToInputShape(signals[sequence], opt.model_name, test_flag=False)
            signal, stage = transformer.ToTensor(signal, stages[sequence], no_cuda=opt.no_cuda)
            out = net(signal)


            loss = criterion(out, stage)
            pred = torch.max(out, 1)[1]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = pred.data.cpu().numpy()
            #pred = pred.data
            stage = stage.data.cpu().numpy()
            #stage = stage.data
            for x in range(len(pred)):
                confusion_mat[stage[x]][pred[x]] += 1
            if i % show_freq == 0:
                plot_result['train'].append(statistics.result(confusion_mat)[3])
                heatmap.draw(confusion_mat, name='train')
                statistics.show(plot_result, epoch + i / (batch_length * 0.95))
                confusion_mat[:] = 0
        if epoch + 1 == opt.epochs:
            plot = 'true'
           
        plot_result, confusion_mat = evalnet(net, signals, stages, test_sequences[fold], epoch + 1, plot, plot_result)
        confusion_mats.append(confusion_mat)
        # scheduler.step()

        if (epoch + 1) % opt.network_save_freq == 0:
            torch.save(net.cpu().state_dict(), './checkpoints/' + opt.model_name + '_epoch' + str(epoch + 1) + '.pth')
            print('network saved.')
            if not opt.no_cuda:
                net.cuda()

        t2 = time.time()
        if epoch + 1 == 1:
            print('cost time: %.2f' % (t2 - t1), 's')

    pos = plot_result['test'].index(min(plot_result['test'])) - 1
    final_confusion_mat += confusion_mats[pos].cpu().numpy()
    util.writelog('fold:' + str(fold + 1) + ' recall,acc,sp,err,k: ' + str(statistics.result(confusion_mats[pos])), True)
    print('------------------')
    util.writelog('confusion_mat:\n' + str(confusion_mats[pos].cpu().numpy()))


util.writelog('final: ' + 'recall,acc,sp,err,k: ' + str(statistics.result(final_confusion_mat)), True)
util.writelog('confusion_mat:\n' + str(final_confusion_mat), True)
statistics.stagefrommat(final_confusion_mat)
heatmap.draw(final_confusion_mat, name='final_test')

epochs_range.close()  # Close the tqdm object

