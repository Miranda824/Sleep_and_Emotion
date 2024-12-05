import numpy as np
import pyeeg as pe
import pickle as pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


channel = [1]  # 14 Channels chosen to fit Emotiv Epoch+
band = [4,8,12,16,25,45]  # 5 bands
window_size = 256  # Averaging band power of 2 sec
step_size = 16  # Each 0.125 sec update once
sample_rate = 128  # Sampling rate of 128 Hz
subjectList = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17',
               '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32']
# List of subjects


# FFT with pyeeg
def FFT_Processing(sub, channel, band, window_size, step_size, sample_rate):
    meta = []

    file_path = "./DEAP/data_preprocessed_python"
    file_name = 's' + sub + '.dat'
    file_full_path = os.path.join(file_path, file_name)

    with open(file_full_path, 'rb') as file:

        subject = pickle.load(file, encoding='latin1')  # resolve the python 2 data problem by encoding : latin1

        for i in range(0, 40):
            # loop over 0-39 trails
            data = subject["data"][i]
            print(data.shape)
            labels = subject["labels"][i]
            # print(labels)
            start = 0

            while start + window_size < data.shape[1]:
                meta_array = []
                meta_data = []  # meta vector for analysis
                for j in channel:
                    X = data[j][start: start + window_size]  # Slice raw data over 2 sec, at interval of 0.125 sec
                    Y = pe.bin_power(X, band, sample_rate)  # FFT over 2 sec of channel j, in seq of theta, alpha, low beta, high beta, gamma
                    meta_data = meta_data + list(Y[0])

                meta_array.append(np.array(meta_data))
                meta_array.append(labels)

                meta.append(np.array(meta_array, dtype=object))
                start = start + step_size
                # print(meta)

        meta = np.array(meta)

        output_dir = "./Emotion_Feature"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'out' + sub + '.npy')
        np.save(output_file, meta, allow_pickle=True, fix_imports=True)

for subjects in subjectList:
    FFT_Processing(subjects, channel, band, window_size, step_size, sample_rate)
