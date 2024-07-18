import os
import time
import random
from scipy.signal import resample
import scipy.io as sio
import numpy as np
import h5py
import mne
import pyeeg as pe
from sklearn.preprocessing import normalize
import transformer
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import load_model

def postprocess_predictions(predictions):
    """
    Round the predicted values to the nearest integer and return as a numpy array.
    """
    rounded_predictions = np.round(predictions)
    return rounded_predictions

def trimdata(data, num):  # Trim the length of data to be a multiple of num
    return data[:num * int(len(data) / num)]

def reducesample(data, mult):  # Downsample the data by a factor of mult
    return data[::mult]

# Delete useless labels
def del_UND(signals, stages):
    stages_copy = stages.copy()
    cnt = 0
    for i in range(len(stages_copy)):
        if stages_copy[i] == 5:
            signals = np.delete(signals, i - cnt, axis=0)
            stages = np.delete(stages, i - cnt, axis=0)
            cnt += 1
    return signals, stages

def connectdata(signal, stage, signals=[], stages=[]):  # Connect new signal and stage data with existing data
    if signals == []:
        signals = signal.copy()
        stages = stage.copy()
    else:
        signals = np.concatenate((signals, signal), axis=0)
        stages = np.concatenate((stages, stage), axis=0)
    return signals, stages

# Load one subject data from cc2018
def loaddata_cc2018(filedir, filename, signal_name, BID, filter=True):
    dirpath = os.path.join(filedir, filename)
    # Load signal
    hea_path = os.path.join(dirpath, os.path.basename(dirpath) + '.hea')
    signal_path = os.path.join(dirpath, os.path.basename(dirpath) + '.mat')

    # Fix path issue
    hea_path = hea_path.replace('\\', '/')
    signal_path = signal_path.replace('\\', '/')

    signal_names = []
    for i, line in enumerate(open(hea_path), 0):
        if i != 0:
            line = line.strip()
            signal_names.append(line.split()[8])
    mat = sio.loadmat(signal_path)
    signals = mat['val'][signal_names.index(signal_name)]

    # Load stage
    stagepath = os.path.join(dirpath, os.path.basename(dirpath) + '-arousal.mat')

    # Fix path issue
    stagepath = stagepath.replace('\\', '/')

    mat = h5py.File(stagepath, 'r')
    # N3(S4+S3)->0  N2->1  N1->2  REM->3  W->4  UND->5
    N3 = mat['data']['sleep_stages']['nonrem3'][0]
    N2 = mat['data']['sleep_stages']['nonrem2'][0]
    N1 = mat['data']['sleep_stages']['nonrem1'][0]
    REM = mat['data']['sleep_stages']['rem'][0]
    W = mat['data']['sleep_stages']['wake'][0]
    UND = mat['data']['sleep_stages']['undefined'][0]
    stages = N3 * 0 + N2 * 1 + N1 * 2 + REM * 3 + W * 4 + UND * 5
    # Resample
    signals = reducesample(signals, 2)
    stages = reducesample(stages, 2)
    # Trim
    signals = trimdata(signals, 3000)
    stages = trimdata(stages, 3000)
    # 30s per label
    signals = signals.reshape(-1, 3000)
    stages = stages[::3000]
    # Balance individualized differences
    signals = transformer.Balance_individualized_differences(signals, BID)
    # Delete UND
    signals, stages = del_UND(signals, stages)

    return signals.astype(np.float16), stages.astype(np.int16)

# Load one subject data from sleep-edfx
# Load data from the specified dataset, signal, number, BID, etc., from the specified folder
# Read the signal data of the specified signal channel in the data file and convert it to numpy array format
# Read the annotation information in the data file and convert the sleep stages to numeric labels
# According to the label information and signal data, segment the sleep signal data into corresponding sleep stages and return the segmented sleep signal data and label data
# Optionally, trim the sleep signal data to retain only the specified time period
# Balance individualized differences in signal data
# Return the processed sleep signal data and label data
def loaddata_sleep_edfx(filedir, filename, signal_name, BID, select_sleep_time, model):
    filenum = filename[2:6]
    filenames = os.listdir(filedir)
    for filename in filenames:
        if str(filenum) in filename and 'Hypnogram' in filename:
            f_stage_name = filename
        if str(filenum) in filename and 'PSG' in filename:
            f_signal_name = filename

    raw_data = mne.io.read_raw_edf(os.path.join(filedir, f_signal_name), preload=True)

    # Implement data preprocessing for emotion classification
    channel_name = "EEG Fpz-Cz"
    channel_index = raw_data.ch_names.index(channel_name)

    # Get the data for the selected channel
    eeg_data = raw_data.get_data(picks=channel_index)
    original_sampling_freq = raw_data.info["sfreq"]
    target_sampling_freq = 128

    # Resample the data to the target sampling frequency (128 Hz)
    num_samples = len(eeg_data[0])
    target_num_samples = int(num_samples * (target_sampling_freq / original_sampling_freq))
    resampled_data = resample(eeg_data[0], target_num_samples)
    num_sequences = 400
    sequence_length = 8064
    num_points_constant = num_sequences * sequence_length
    reshaped_data = resampled_data[:num_points_constant].reshape(num_sequences, sequence_length)
    channel = [1]
    band = [4, 8, 12, 16, 25, 45]
    window_size = 256  # Averaging band power of 2 sec
    step_size = 16  # Each 0.125 sec update once
    sample_rate = 128  # Sampling rate of 128 Hz

    meta = []
    data = reshaped_data
    start = 0
    while start + window_size < data.shape[1]:
        meta_array = []
        meta_data = []  # Meta vector for analysis
        for j in channel:
            X = data[j][start: start + window_size]  # Slice raw data over 2 sec, at interval of 0.125 sec
            Y = pe.bin_power(X, band, sample_rate)  # FFT over 2 sec of channel j, in seq of theta, alpha, low beta, high beta, gamma
            meta_data = meta_data + list(Y[0])

        meta_array.append(np.array(meta_data))
        meta_array.append([1, 1, 1, 1])

        meta.append(np.array(meta_array, dtype=object))
        start = start + step_size

    meta = np.array(meta)
    unlabeled_data = meta

    data_training = []
    for i in range(0, unlabeled_data.shape[0]):
        data_training.append(unlabeled_data[i][0])

    X = np.array(data_training)
    X = normalize(X)  # Normalize

    X_training = X
    X_scaled_training = pd.DataFrame(data=X_training).values
    X_scaled_training = tf.convert_to_tensor(X_scaled_training, dtype=tf.float32)

    predictions = model.predict(X_scaled_training.numpy())
    processed_predictions = predictions
    print('look here:', processed_predictions.shape)
    emo_data, emo_index = np.max(processed_predictions, axis=1), np.argmax(processed_predictions, axis=1)
    print(emo_index.shape)

    raw_annot = mne.read_annotations(os.path.join(filedir, f_stage_name))

    eeg = raw_data.pick_channels([signal_name]).to_data_frame().values.T
    eeg = eeg[1]
    eeg = eeg.reshape(-1)

    raw_data.set_annotations(raw_annot, emit_warning=False)
    # N3(S4+S3)->0  N2->1  N1->2  REM->3  W->4  other->UND->5
    event_id = {'Sleep stage 4': 0,
                'Sleep stage 3': 0,
                'Sleep stage 2': 1,
                'Sleep stage 1': 2,
                'Sleep stage R': 3,
                'Sleep stage W': 4,
                'Sleep stage ?': 5,
                'Movement time': 5}
    events, event_dict = mne.events_from_annotations(
        raw_data, event_id=event_id, chunk_duration=30.)

    stages = []
    signals = []
    for i in range(len(events) - 1):
        stages.append(events[i][2])
        signals.append(eeg[events[i][0]:events[i][0] + 3000])
    stages = np.array(stages)
    signals = np.array(signals)

    # #select sleep time
    if select_sleep_time:
        if 'SC' in f_signal_name:
            # first_annot = raw_annot[0]
            # duration_80percent = int(first_annot['duration'] * 0.2)
            # first_annot['duration'] = duration_80percent
            # raw_annot[0] = first_annot

            signals = signals[np.clip(int(raw_annot[0]['duration'])//30-60,0,9999999):int(raw_annot[-2]['onset'])//30+60]
            stages = stages[np.clip(int(raw_annot[0]['duration'])//30-60,0,9999999):int(raw_annot[-2]['onset'])//30+60]
            # signals = signals[np.clip(int(raw_annot[0]['duration'])//30-60,0,9999999):int(raw_annot[-2]['onset'])//30+60]
            # stages = stages[np.clip(int(raw_annot[0]['duration'])//30-60,0,9999999):int(raw_annot[-2]['onset'])//30+60]
    # if select_sleep_time:
    #     if 'SC' in f_signal_name:
    #         duration = int(raw_annot[0]['duration'])
    #         onset = int(raw_annot[0]['onset'])
    #         remove_end = int(onset) + duration - (500 * 100)
    #         signals = signals[np.clip(int(remove_end) // 30 - 60, 0, 9999999):int(raw_annot[-2]['onset']) // 30 + 60]
    #         stages = stages[np.clip(int(remove_end) //30-60,0,9999999):int(raw_annot[-2]['onset'])//30+60]

    signals,stages = del_UND(signals, stages)
    # print('shape:',signals.shape,stages.shape)

    signals = transformer.Balance_individualized_differences(signals, BID)
    # Connect the emotion classification results to the original input signal of sleep stage 
    # hard-code the classification results into all segments, each segment has a shape of 3000 in the original sleep stage input, and now it becomes 3488
    print('all_1:{}'.format((emo_index==1).all()))
    emo_index = np.repeat(emo_index.reshape(1,-1),signals.shape[0],axis=0)
    # print(emo_index.shape)


    signals = np.concatenate([signals,emo_index],axis=1)

    import matplotlib.pyplot as plt
    plt.plot(np.arange(signals.shape[-1]),np.mean(signals,axis=0))
    data = np.column_stack((np.arange(signals.shape[-1]), np.mean(signals,axis=0)))

    np.save('coordinates', data)
    # plt.show()

    return signals.astype(np.float16),stages.astype(np.int16)

#load all data in datasets
def loaddataset(filedir,dataset_name,signal_name,num,BID,select_sleep_time,shuffle = True,model=None):#num，要加载的数据数量

    print('load dataset, please wait...')
    filenames = os.listdir(filedir)
    if shuffle:  # Get a list of file names for all files in a folder. If shuffle is True, the order of the file name list will be shuffled.
        random.shuffle(filenames)
    signals=[]
    stages=[]
    if dataset_name in ['sleep-edfx','sleep-edfx-8']:
        # if num > 197:
        #      num = 197
        if dataset_name == 'sleep-edfx-8':
            filenames = ['SC4002E0-PSG.edf','SC4012E0-PSG.edf','SC4102E0-PSG.edf','SC4112E0-PSG.edf',
            'ST7022J0-PSG.edf','ST7052J0-PSG.edf','ST7121J0-PSG.edf','ST7132J0-PSG.edf']
        cnt = 0
        for filename in filenames:
            if 'PSG' in filename:
                signal,stage = loaddata_sleep_edfx(filedir,filename,signal_name,BID,select_sleep_time,model)
                signals,stages = connectdata(signal,stage,signals,stages)
                # cnt += 1
                # if cnt == num:
                #     break
    return signals,stages



