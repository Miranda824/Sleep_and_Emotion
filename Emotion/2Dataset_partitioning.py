import os
import numpy as np
import time

subjectList = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17',
               '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32']

# Load the .npy file
file_path = os.path.join(os.getcwd(), 'Emotion Feature/out01.npy')
data = np.load(file_path, allow_pickle=True)

# Print the dimensions (shape) of the data
print("Data dimensions (shape):", data.shape)
# Print the entire data
print("Data:", data)

data_training = []
label_training = []
data_testing = []
label_testing = []
data_validation = []
label_validation = []

for subjects in subjectList:
    file_path = "/home/ti80/Documents/github/Sleep and Emotion/Emotion/Emotion Feature"
    file_name = 'out' + subjects + '.npy'
    file_full_path = os.path.join(file_path, file_name)

    with open(file_full_path, 'rb') as file:
        sub = np.load(file, allow_pickle=True)
        print(sub.shape)
        for i in range(0, sub.shape[0]):
            if i % 8 == 0:
                data_testing.append(sub[i][0])
                print(sub[i][0])
                time.sleep(10)
                label_testing.append(sub[i][1])
            elif i % 8 == 1:
                data_validation.append(sub[i][0])
                label_validation.append(sub[i][1])
            else:
                data_training.append(sub[i][0])
                label_training.append(sub[i][1])

np.save('data_training', np.array(data_training), allow_pickle=True, fix_imports=True)
np.save('label_training', np.array(label_training), allow_pickle=True, fix_imports=True)
print("training dataset:", np.array(data_training).shape, np.array(label_training).shape)

np.save('data_testing', np.array(data_testing), allow_pickle=True, fix_imports=True)
np.save('label_testing', np.array(label_testing), allow_pickle=True, fix_imports=True)
print("testing dataset:", np.array(data_testing).shape, np.array(label_testing).shape)

np.save('data_validation', np.array(data_validation), allow_pickle=True, fix_imports=True)
np.save('label_validation', np.array(label_validation), allow_pickle=True, fix_imports=True)
print("validation dataset:", np.array(data_validation).shape, np.array(label_validation).shape)
