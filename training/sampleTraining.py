import tensorflow as tf
import numpy as np
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.models import load_model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Masking
from keras.layers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.models import model_from_json
from sklearn.utils import shuffle
from keras.optimizers import rmsprop
import json
import pickle
import time
import gc
import os

# ArrayList
test_acc = []
X_test_final = []
y_test_final = []
# Filepath's
input_traindata_path = "D:/datasets/lowquality_wordalignment/speaker_input_train"
output_traindata_path = "D:/datasets/lowquality_wordalignment/speaker_final_output_train"
input_testdata_path = "D:/datasets/lowquality_wordalignment/speaker_input_test"
output_testdata_path = "D:/datasets/lowquality_wordalignment/speaker_final_output_test"

checkpoint = "checkpoint.ckpt"
checkpoint_dir = os.path.dirname(checkpoint)

# instantiate Tensorboard
NAME = "Lip-Reading-{}".format(int(time.time()))
Tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

# divide usage of gpu memory
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


def save_model(model, save_topo_to):
    model_json = model.to_json()
    with open(save_topo_to, 'w') as json_file:
        json_file.write(model_json)
    del model

def create_model(max_seqlen=40, image_size=(40, 40), fc_size=128, save_topo_to='structure.json', save_result=True):
    model = Sequential()

    print("Adding TimeDistributeDense Layer...")
    model.add(Dense(fc_size, input_shape=(max_seqlen, image_size[0] * image_size[1])))

    print("Adding Masking Layer...")
    model.add(Masking(mask_value=0.0))

    print("Adding First LSTM Layer...")
    model.add(LSTM(fc_size, return_sequences=True))

    print("Adding Second LSTM Layer...")
    model.add(LSTM(fc_size, return_sequences=False))

    print("Adding Final Dense Layer...")
    model.add(Dense(52))

    print("Adding Softmax Layer...")
    model.add(Activation('softmax'))

    rmsprop(lr=0.0002, rho=0.9, epsilon=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    if save_result:
        print("Saving Model to file...")
        save_model(model, save_topo_to)
    return model


def generate_batches(filesX, fileY):
    counter = 0
    while True:

        if counter == len(filesX):
            counter = 0

        filx = filesX[counter]
        fily = fileY[counter]

        fill = np.load(filx)

        X_train = fill['arr_0']

        y_train = np.load(fily)

        counter += 1

        yield X_train, y_train

checkpoint = ModelCheckpoint(checkpoint, monitor='loss', verbose=1,
                             save_weights_only=True, save_best_only=False, mode='min')

model = create_model()

traindata = []
outputTrain = []
testdata = []
outputTest = []
for x in range(1,33):
    traindata.append(input_traindata_path+str(x)+'.npz')
    outputTrain.append(output_traindata_path+str(x)+'.npy')

    testdata.append(input_testdata_path+str(x)+'.npz')
    outputTest.append(output_testdata_path+str(x)+'.npy')


with tf.device('/cpu:0'):
    model.fit_generator(generate_batches(traindata, outputTrain),
                        4, epochs=2, callbacks=[Tensorboard, checkpoint])

# X_test_final = np.array(X_test_final)
# y_test_final = np.array(y_test_final)
    score, acc = model.evaluate_generator(generate_batches(testdata, outputTest), 4)

