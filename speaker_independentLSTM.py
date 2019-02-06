import lipreadtrain
import gc
import numpy as np
import tensorflow as tf
import os
import time
from keras.callbacks import TensorBoard,ModelCheckpoint
from sklearn.utils import shuffle



checkpoint = "training/checkpoint.ckpt"
checkpoint_dir = os.path.dirname(checkpoint)

#instantiate Tensorboard
NAME = "Lip-Reading-{}".format(int(time.time()))
Tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

#divide usage of gpu memory
gpu_options  = tf.GPUOptions(per_process_gpu_memory_fraction=0.222)
sess = tf.Session(config=tf.ConfigProto(gpu_options = gpu_options))

model = lipreadtrain.build_network(dict_size=52,
                                lr=0.0002,
                                max_seqlen=40,
                                image_size=(40,40),
                                optimizer='rmsprop',load_cache=True)



test_acc = []
X_test_final = []
y_test_final = []
input_traindata_path = "D:/datasets/lowquality_wordalignment/speaker_input_train"
output_traindata_path = "D:/datasets/lowquality_wordalignment/speaker_final_output_train"
input_testdata_path = "D:/datasets/lowquality_wordalignment/speaker_input_test"
output_testdata_path = "D:/datasets/lowquality_wordalignment/speaker_final_output_test"


for speaker_id in range(1, 33):
	model.load_weights(checkpoint)

	fill = np.load(input_traindata_path + str(speaker_id)+".npz")
	X_train = fill['arr_0']
	print(X_train)
	y_train = np.load(output_traindata_path + str(speaker_id)+".npy")
	X_train, y_train = shuffle(X_train, y_train, random_state=0)

	fil2 = np.load(input_testdata_path + str(speaker_id) + ".npz")
	X_test = fil2['arr_0']
	y_test = np.load(output_testdata_path + str(speaker_id) + ".npy")

	checkpoint = ModelCheckpoint(checkpoint, monitor='loss', verbose=1,save_weights_only=True,
								save_best_only=True, mode='max', period=5)

	model.fit(X_train, y_train, batch_size=100, epoch=5,
				validation_split=0.2, callbacks=[TensorBoard, checkpoint])

	X_test_final = X_test_final + list(X_test)
	y_test_final = y_test_final + list(y_test)
	fill.close()
	gc.collect()

X_test_final = np.array(X_test_final)
y_test_final = np.array(y_test_final)
score,acc = model.evaluate(X_test_final, y_test_final)

print("Test accuracy ",acc," Test score ",score)
