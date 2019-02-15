import numpy as np


input_traindata_path = "D:/datasets/lowquality_wordalignment/speaker_input_train"
input_testdata_path = "D:/datasets/lowquality_wordalignment/speaker_input_test"

for x in range(1,4):

    path =input_testdata_path + str(x)

    fill = np.load(path + ".npz")
    fill = fill['arr_0']

    print(len(fill))
    np.save(path+'.npy',fill)