import tensorflow as tf
import numpy as np
np.random.seed(1337)
from keras.callbacks import TensorBoard,ModelCheckpoint
from keras.models import load_model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Masking
from keras.layers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.models import model_from_json
from sklearn.utils import shuffle
from keras.optimizers import rmsprop
import pickle
import time
import gc
import os

#ArrayList
test_acc = []
X_test_final = []
y_test_final = []
#Filepath's
input_traindata_path = "D:/datasets/lowquality_wordalignment/speaker_input_train"
output_traindata_path = "D:/datasets/lowquality_wordalignment/speaker_final_output_train"
input_testdata_path = "D:/datasets/lowquality_wordalignment/speaker_input_test"
output_testdata_path = "D:/datasets/lowquality_wordalignment/speaker_final_output_test"



#instantiate Tensorboard
NAME = "Lip-Reading"
tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

#divide usage of gpu memory
gpu_options  = tf.GPUOptions(per_process_gpu_memory_fraction=0.222)
sess = tf.Session(config=tf.ConfigProto(gpu_options = gpu_options))

for speaker_id in range(1,33):
    model = load_model('model.h5')
    fil = np.load(input_traindata_path + str(speaker_id)+".npz")
    X_train = fil['arr_0']
    y_train = np.load(output_traindata_path + str(speaker_id)+".npy")
    X_train, y_train = shuffle(X_train, y_train, random_state=0)
    fil2 = np.load(input_testdata_path + str(speaker_id) + ".npz")
    X_test = fil2['arr_0']
    y_test = np.load(output_testdata_path + str(speaker_id) + ".npy")

    model.fit(X_train, y_train, batch_size=100, nb_epoch=10, validation_split=0.4, callbacks=[tensorboard])
    X_test_final = X_test_final + list(X_test)
    y_test_final = y_test_final + list(y_test)
    model.save('model.h5',overwrite=True,include_optimizer=True)
    del model
    del X_train
    del y_train
    fil.close()
    gc.collect()

X_test_final = np.array(X_test_final)
y_test_final = np.array(y_test_final)

model = load_model('model.h5')
score,acc = model.evaluate(X_test_final,y_test_final)
print("Test accuracy ",acc," Test score ",score)