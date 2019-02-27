import numpy as np


input_traindata_path = "D:/Datasets/output/speaker_input_train"
input_testdata_path = "D:/Datasets/output/speaker_input_test"

path =input_traindata_path + str(1)

fill = np.load(path + ".npz")
fill = fill['arr_0']

print(np.shape(fill))
