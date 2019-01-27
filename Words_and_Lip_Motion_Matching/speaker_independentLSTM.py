import lipreadtrain
import gc
import numpy as np
from copy import copy
from sklearn.utils import shuffle
import pickle

net = lipreadtrain.build_network(dict_size=52,
                                lr=0.0002,
                                max_seqlen=40,
                                image_size=(40,40),
                                optimizer='rmsprop',load_cache=False)

test_acc = []
X_test_final = []
y_test_final = []
input_traindata_path = "D:/datasets/lowquality_wordalignment/speaker_input_train"
output_traindata_path = "D:/datasets/lowquality_wordalignment/speaker_final_output_train"
input_testdata_path = "D:/datasets/lowquality_wordalignment/speaker_input_test"
output_testdata_path = "D:/datasets/lowquality_wordalignment/speaker_final_output_test"


for speaker_id in range(1, 33):
	for j in range(0,12):
		fill = np.load(input_traindata_path + str(speaker_id)+".npz")
		X_train = fill['arr_0']
		y_train = np.load(output_traindata_path + str(speaker_id)+".npy")
		#y_train = y_train[j*500:(j+1)*500]
		X_train, y_train = shuffle(X_train, y_train, random_state=0)

		fil2 = np.load(input_testdata_path + str(speaker_id) + ".npz")
		X_test = fil2['arr_0']
		y_test = np.load(output_testdata_path + str(speaker_id) + ".npy")

		net.fit(X_train, y_train, batch_size=100, nb_epoch=100,
				validation_split=0.0)
		X_test_final = X_test_final + list(X_test)
		y_test_final = y_test_final + list(y_test)
		del X_train
		del y_train
		fill.close()
		gc.collect()
X_test_final = np.array(X_test_final)
y_test_final = np.array(y_test_final)
score,acc = net.evaluate(X_test_final,y_test_final,show_accuracy=True)
print("Test accuracy ",acc," Test score ",score)
