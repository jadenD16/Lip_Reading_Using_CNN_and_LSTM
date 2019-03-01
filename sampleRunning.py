from imutils import face_utils
import dlib
import cv2
import numpy as np
import json
from keras.layers.wrappers import *
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Convolution1D
from keras.optimizers import *
from keras.datasets import imdb
from keras.callbacks import EarlyStopping, TensorBoard
from keras.models import load_model
from keras.models import model_from_json
from sklearn.utils import shuffle
import tensorflow as tf
from keras.models import Sequential
from keras.models import load_model
from keras.layers.core import Dense, Dropout, Activation, Masking
from keras.layers import TimeDistributed
import gc
from keras.optimizers import rmsprop



# p = our pre-treined model directory, on my case, it's on the same script's diretory.
p = "C:\\Users\\Jaden\\Downloads\\shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

counter = 0

path = 'C:/Users/Jaden/PycharmProjects/Thesis/Lip_Reading_Using_CNN_and_LSTM/training/model'

def normalizedMouth(mouth):

    image_min = mouth[mouth > 0].min()
    image_max = mouth[mouth > 0].max()

    mouth = (mouth - image_min) / (float(image_max - image_min))

    return mouth

def read_model(weights_filename='untrained_weight.h5',
               mod='model-2.h5'):

    global path

    print("Reading Model from " + weights_filename + " and " + mod)
    print("Please wait, it takes time.")
    model = load_model(path+'/'+mod)
    #model.load_weights(weights_filename)
    print("Finish Reading!")
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

cap = cv2.VideoCapture(path)
x2 = 0
y2 = 0
w2 = 0
h2 = 0
counter = 0

wordSequence = np.zeros((40, 1600))
wCounter = 0

model = read_model()

while True:
    # Getting out image by webcam
    _, image = cap.read()

    image = cv2.resize(image,(360,288))

    # Converting the image to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Get faces into webcam's image
    rects = detector(gray, 0)
    mask = np.zeros(image.shape, dtype="uint8")

    counter+= 1
    time = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    if counter == time:
            break

    # For each detected face, find the landmark.
    for (i, rect) in enumerate(rects):
    # Make the prediction and transfom it to numpy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)


        if counter == 1:
            minn = np.asarray(shape[48:68])

            min2 = minn.min(0)
            max2 = minn.max(0)

            x2, y2 = min2
            w2, h2 = max2

            y2 -=7
            h2 +=4

            cropped = image[y2:h2,x2:w2]
            try:
                #resize the crop image to 40x40 dimension
                croppedMouth = cv2.resize(cropped,(40,40))
            except cv2.error:
                continue
            #normalized Mouth coordinates
            normMouth = normalizedMouth(croppedMouth)

            normMouth = np.array(normMouth[:, 0:, :2])

            normMouth = normMouth.transpose(2, 0, 1).reshape(40,-1)

            normMouth = np.resize(normMouth, 1600)

            wordSequence[wCounter] = normMouth
            wCounter += 1

            model.predict(wordSequence)
                    #wCounter = 0
                    #sentenceList.append(wordSequence)
                    #wordSequence = np.zeros((40,1600))
                    #wordSequence[wCounter] = normMouth

    cv2.waitKey(1)
cap.release()
