import numpy as np
import torch


def trimdata(data,num):
    return data[:num*int(len(data)/num)]

def shuffledata(data,target):
    state = np.random.get_state()
    np.random.shuffle(data)
    np.random.set_state(state)
    np.random.shuffle(target)

def batch_generator_subject(data,target,batchsize,shuffle = True):
    data_test = data[int(0.95*len(target)):]
    data_train = data[0:int(0.95*len(target))]
    target_test = target[int(0.95*len(target)):]
    target_train = target[0:int(0.95*len(target))]
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
    target = target.reshape(-1,batchsize)
    return data,target

def k_fold_generator(length,fold_num):# Generate training and test sequences for each fold in cross-validation.
    # If fold_num is 1, use the first 95% as the training set and the last 5% as the test set;
    # Otherwise, split the sequence into fold_num parts, each part is used as the test set once, and the rest as the training set.
    sequence = np.linspace(0,length-1,length,dtype='int')
    if fold_num == 1:
        train_sequence = sequence[0:int(0.95*length)].reshape(1,-1)
        test_sequence = sequence[int(0.95*length):].reshape(1,-1)
    else:
        train_length = int(length/fold_num*(fold_num-1))
        test_length = int(length/fold_num)
        train_sequence = np.zeros((fold_num,train_length), dtype = 'int')
        test_sequence = np.zeros((fold_num,test_length), dtype = 'int')
        for i in range(fold_num):
            test_sequence[i] = (sequence[test_length*i:test_length*(i+1)])[:test_length]
            train_sequence[i] = np.concatenate((sequence[0:test_length*i],sequence[test_length*(i+1):]),axis=0)[:train_length]

    return train_sequence,test_sequence

def Normalize(data,maxmin,avg,sigma):
    data = np.clip(data, -maxmin, maxmin)
    return (data-avg)/sigma

def Balance_individualized_differences(signals,BID):# This function normalizes the input signals to balance individual differences.
    # signals is the array of input signals, BID is a string used to determine the type of normalization.
    if BID == 'median':
        signals = (signals*8/(np.median(abs(signals))))
        signals=Normalize(signals,maxmin=10e3,avg=0,sigma=30)
    elif BID == '5_95_th':
        tmp = np.sort(signals.reshape(-1))
        th_5 = -tmp[int(0.05*len(tmp))]
        signals=Normalize(signals,maxmin=10e3,avg=0,sigma=th_5)
    elif BID == 'zscore':
        # Ensures that the signals are normalized to have a mean of 0 and a standard deviation of 1.
        signals = (signals - np.mean(signals)) / np.std(signals)
        signals = Normalize(signals, maxmin=10e3, avg=0, sigma=30)
    else:
        signals = Normalize(signals, maxmin=10e3, avg=0, sigma=30)
    return signals

def ToTensor(data,target=None,no_cuda = False):# Function to convert Numpy arrays to PyTorch tensors
    if target is not None:
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

def ToInputShape(data):
    data = data.astype(np.float32)
    batchsize=data.shape[0]
    result = []

    for i in range(0,batchsize):
        result.append(data[i][:3000])
    result = np.array(result)
    result = result.reshape(batchsize,1,3000)
    result = np.concatenate([result,data[:,3000:].reshape(batchsize,1,-1)],axis=2)
    result=torch.from_numpy(result)

    return result.to('cuda:0')
