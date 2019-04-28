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
               model='model2.h5'):

    global path

    print("Reading Model from " + model)
    print("Please wait, it takes time.")
    model = load_model(model)
    #model.load_weights('C:/Users/Jaden/Downloads/GRIDcorpus-experiments-master/TRAINED-MODELS/weightts.hdf5')
    print("Finish Reading!")
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


position = 0

def putSub(frame,word):

    global position

    font = cv2.FONT_HERSHEY_SIMPLEX

    #position = (len(str(word)) + 5) + 280

    cv2.putText(image, word, (280,220), font, 1, (0, 0, 0), 2)

    return image



vidFile = 'D:/Datasets/s1/lwir2n.mpg'
cap = cv2.VideoCapture(vidFile)
x2 = 0
y2 = 0
w2 = 0
h2 = 0
counter = 0

wordSequence = np.zeros((1,40, 1600))
wCounter = 0

model = read_model()
previous = 0
wordDict = {'set': 36, 'i': 17, 'b': 3, 'u': 43, 'green': 15, 'five': 12, 'v': 44, 'blue': 5, 'm': 23, 'seven': 37, 'e': 9, 'red': 34, 'z': 49, 'place': 30, 'r': 33, 'q': 32, 'one': 28, 'at': 2, 'bin': 4, 'f': 11, 'with': 46, 'x': 47, 'now': 26, 'eight': 10, 'a': 0, 'please': 31, 'soon': 39, 'by': 6, 'in': 18, 't': 40, 'g': 14, 'c': 7, 'four': 13, 'zero': 50, 'o': 27, 'lay': 22, 'p': 29, 'again': 1, 'j': 19, 'n': 24, 'six': 38, 'two': 42, 'l': 21, 'd': 8, 'three': 41, 'h': 16, 'white': 45, 'nine': 25, 'y': 48, 'k': 20, 's': 35}


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

        wordSequence[0][wCounter] = normMouth
        wCounter += 1



        if wCounter == 8:
            a = model.predict_classes(wordSequence)
            if previous != a[0]:
                word = list(wordDict.keys())[list(wordDict.values()).index(a[0])]
                print(word)
                image = putSub(image, word)

            wordSequence = np.zeros((1, 40, 1600))



            wCounter = 0
            previous = a[0]

    cv2.imshow('lo;',image)
    cv2.waitKey(1)
cap.release()
