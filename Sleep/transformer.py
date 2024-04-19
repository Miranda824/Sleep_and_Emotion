import os
import random
import numpy as np
import torch


def trimdata(data,num):
    return data[:num*int(len(data)/num)]

def shuffledata(data,target):
    state = np.random.get_state()
    np.random.shuffle(data)
    np.random.set_state(state)
    np.random.shuffle(target)
    # return data,target

def batch_generator_subject(data,target,batchsize,shuffle = True):
    data_test = data[int(0.8*len(target)):]
    data_train = data[0:int(0.8*len(target))]
    target_test = target[int(0.8*len(target)):]
    target_train = target[0:int(0.8*len(target))]
    data_test,target_test = batch_generator(data_test, target_test, batchsize)
    data_train,target_train = batch_generator(data_train, target_train, batchsize)
    data = np.concatenate((data_train, data_test), axis=0)
    target = np.concatenate((target_train, target_test), axis=0)
    return data,target

def batch_generator(data,target,batchsize,shuffle = True):
    if shuffle:
        shuffledata(data,target)
    data = trimdata(data,batchsize)
    target = trimdata(target,batchsize)
    data = data.reshape(-1,batchsize,3488)
    # data = data.reshape(-1, batchsize, 3000)
    target = target.reshape(-1,batchsize)
    return data,target

def k_fold_generator(length,fold_num):#生成交叉验证中每一折的训练集和测试集序列下标。
    # 如果折数为1，则将前80%作为训练集，后20%作为测试集；
    # 否则将序列分成fold_num份，依次取其中一份作为测试集，其余作为训练集。
    sequence = np.linspace(0,length-1,length,dtype='int')
    if fold_num == 1:
        train_sequence = sequence[0:int(0.8*length)].reshape(1,-1)
        test_sequence = sequence[int(0.8*length):].reshape(1,-1)
    else:
        train_length = int(length/fold_num*(fold_num-1))
        test_length = int(length/fold_num)
        train_sequence = np.zeros((fold_num,train_length), dtype = 'int')
        test_sequence = np.zeros((fold_num,test_length), dtype = 'int')
        for i in range(fold_num):
            test_sequence[i] = (sequence[test_length*i:test_length*(i+1)])[:test_length]
            train_sequence[i] = np.concatenate((sequence[0:test_length*i],sequence[test_length*(i+1):]),axis=0)[:train_length]

    return train_sequence,test_sequence

# def k_fold_generator(length, fold_num):
#     # 如果折数为1，则将前80%作为训练集，后10%作为验证集，后10%作为测试集；
#     # 否则将序列分成fold_num份，依次取其中一份作为测试集，剩余部分平均分成两部分作为训练集和验证集。
#     sequence = np.linspace(0, length - 1, length, dtype='int')
#     if fold_num == 1:
#         train_sequence = sequence[0:int(0.8 * length)].reshape(1, -1)
#         val_sequence = sequence[int(0.8 * length):int(0.9 * length)].reshape(1, -1)
#         test_sequence = sequence[int(0.9 * length):].reshape(1, -1)
#     else:
#         train_length = int(length / fold_num * (fold_num - 2))
#         val_length = int(length / fold_num)
#         test_length = int(length / fold_num)
#         train_sequence = np.zeros((fold_num, train_length), dtype='int')
#         val_sequence = np.zeros((fold_num, val_length), dtype='int')
#         test_sequence = np.zeros((fold_num, test_length), dtype='int')
#         for i in range(fold_num):
#             test_sequence[i] = sequence[test_length * i:test_length * (i + 1)]
#             val_sequence[i] = sequence[train_length + i * val_length:train_length + (i + 1) * val_length]
#             train_sequence[i] = np.concatenate((sequence[0:train_length + i * val_length],
#                                                 sequence[train_length + (i + 1) * val_length:]), axis=0)
#
#     return train_sequence, val_sequence, test_sequence


def Normalize(data,maxmin,avg,sigma):
    data = np.clip(data, -maxmin, maxmin)
    return (data-avg)/sigma

def Balance_individualized_differences(signals,BID):#这个函数实现了对输入信号进行归一化的功能，以平衡信号之间的个体差异
    # signals表示输入信号的数组，BID为字符串，用于确定信号归一化的类型。

    if BID == 'median':
        signals = (signals*8/(np.median(abs(signals))))
        signals=Normalize(signals,maxmin=10e3,avg=0,sigma=30)
    elif BID == '5_95_th':
        tmp = np.sort(signals.reshape(-1))
        th_5 = -tmp[int(0.05*len(tmp))]
        signals=Normalize(signals,maxmin=10e3,avg=0,sigma=th_5)
    elif BID == 'zscore':
        # 确保了信号被归一化为均值为 0 且标准差为 1。
        signals = (signals - np.mean(signals)) / np.std(signals)
        signals = Normalize(signals, maxmin=10e3, avg=0, sigma=30)
    else:
        signals = Normalize(signals, maxmin=10e3, avg=0, sigma=30)
    # else :
    #     #dataser 5_95_th  median
    #     #CC2018  24.75   7.438
    #     #sleep edfx  37.4   9.71
    #     #sleep edfx sleeptime  39.03   10.125
    #
    #     signals = Normalize(signals, maxmin=10e3, avg=0, sigma=30)

        # signals=Normalize(signals,maxmin=10e3,avg=0,sigma=30)
    return signals

def ToTensor(data,target=None,no_cuda = False):#将Numpy数组转换为PyTorch张量的函数
    # print('data:',data.shape)
    if target is not None:
        # data = torch.from_numpy(data).float()
        data = torch.Tensor(data).float()
        target = torch.from_numpy(target).long()
        if not no_cuda:
            data = data.cuda()
            target = target.cuda()
        return data,target
    else:
        data = torch.from_numpy(data).float()
        if not no_cuda:
            data = data.cuda()
        return data

def random_transform_1d(data,finesize,test_flag):#对1D数据进行随机变换
    length = len(data)
    if test_flag:
        move = int((length-finesize)*0.5)
        result = data[move:move+finesize]
    else:
        #random crop    
        move = int((length-finesize)*random.random())
        result = data[move:move+finesize]
        #random flip
        if random.random()<0.5:
            result = result[::-1]
        #random amp
        result = result*random.uniform(0.8,1.2)

    return result

def random_transform_2d(img,finesize = (224,122),test_flag = True):
    h,w = img.shape[:2]
    if test_flag:
        h_move = 2
        w_move = int((w-finesize[1])*0.5)
        result = img[h_move:h_move+finesize[0],w_move:w_move+finesize[1]]
    else:
        #random crop
        h_move = int(10*random.random()) #do not loss low freq signal infos
        w_move = int((w-finesize[1])*random.random())
        result = img[h_move:h_move+finesize[0],w_move:w_move+finesize[1]]
        #random flip
        if random.random()<0.5:
            result = result[:,::-1]
        #random amp
        result = result*random.uniform(0.9,1.1)+random.uniform(-0.05,0.05)
    return result

def ToInputShape(data,net_name,test_flag = False):
    data = data.astype(np.float32)
    batchsize=data.shape[0]
    if net_name in['attNsoft1']:
        result =[]
        for i in range(0,batchsize):
            randomdata=random_transform_1d(data[i][:3000],finesize = 2700,test_flag=test_flag)
            result.append(randomdata)
        result = np.array(result)
        result = result.reshape(batchsize,1,2700)
        result = np.concatenate([result,data[:,3000:].reshape(batchsize,1,-1)],axis=2)
        result=torch.from_numpy(result)

    return result.to('cuda:0')
