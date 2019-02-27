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
import gc
from keras.optimizers import rmsprop

#ArrayList
test_acc = []
X_test_final = []
y_test_final = []
#Filepath's
input_traindata_path = "D:/Datasets/output/speaker_input_train"
output_traindata_path = "D:/Datasets/output/speaker_final_output_train"
input_testdata_path = "D:/Datasets/output/speaker_input_test"
output_testdata_path = "D:/Datasets/output/speaker_final_output_test"


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
    model.add(LSTM(fc_size, return_sequences=True))

    print("Adding Final Dense Layer...")
    model.add(Dense(52))

    print("Adding Softmax Layer...")
    model.add(Activation('softmax'))


    rmsprop(lr=0.0002, rho=0.9, epsilon=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    if save_result:
        print("Saving Model to file...")
        print("saved Model")
        save_model(model, save_topo_to)
    return model


#instantiate Tensorboard
NAME = "Lip-Reading"
tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

model = create_model()
model.save('model.h5')

#divide usage of gpu memory
gpu_options  = tf.GPUOptions(per_process_gpu_memory_fraction=0.222)
sess = tf.Session(config=tf.ConfigProto(gpu_options = gpu_options))

for speaker_id in range(1,4):
    model = load_model('model.h5')
    fil = np.load(input_traindata_path + str(speaker_id)+".npz")
    X_train = fil['arr_0']
    y_train = np.load(output_traindata_path + str(speaker_id)+".npy")
    X_train, y_train = shuffle(X_train, y_train, random_state=0)
    fil2 = np.load(input_testdata_path + str(speaker_id) + ".npz")
    X_test = fil2['arr_0']
    y_test = np.load(output_testdata_path + str(speaker_id) + ".npy")

    model.fit(X_train, y_train, batch_size=100, nb_epoch=4, validation_split=0.4, callbacks=[tensorboard])
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