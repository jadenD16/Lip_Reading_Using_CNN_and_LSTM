import tflearn
from keras.models import  load_model
from keras.models import Sequential
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import tensorflow as tf
import matplotlib as plt
from pycm import ConfusionMatrix
from  keras.utils import  plot_model


input_testdata_path = "D:/Datasets/output/speaker_input_tests"
output_testdata_path = "D:/Datasets/output/speaker_final_output_tests"

def read_model(weights_filename='untrained_weight.h5',
               model='model2.h5'):

    global path

    print("Reading Model from " + model)
    print("Please wait, it takes time.")
    model = load_model(model)
    #model.load_weights('C:/Users/Jaden/Downloads/GRIDcorpus-experiments-master/TRAINED-MODELS/weightts.hdf5')
    print("Finish Reading!")
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


model = read_model()


speaker_id = 5

X_test = np.load(input_testdata_path + "610" + ".npz")
y_test = np.load(output_testdata_path + "610" + ".npy")

X_test = X_test['arr_0']

print(np.shape(X_test))

data=model.predict_classes(X_test)
#print(np.shape(data))
#print(np.shape(y_test))

a,b = np.where(y_test == 1)

actual_array = np.empty(260)
predicted_array = np.empty(260)

actual_array = b
predicted_array = data


plot_model(model, show_shapes=True,)

# Plot non-normalized confusion matrix
cm = ConfusionMatrix(actual_array,predicted_array)

print(cm)

cm.save_csv('confusion_matrix')