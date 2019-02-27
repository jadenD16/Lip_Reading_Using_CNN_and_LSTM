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

# p = our pre-treined model directory, on my case, it's on the same script's diretory.
p = "C:\\Users\\Jaden\\Downloads\\shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

cap = cv2.VideoCapture('D:\\Datasets\\s1\\bbaf2n.mpg')

counter = 0

weight,height = (40,40)

path = 'C:/Users/Jaden/PycharmProjects/Thesis/Lip_Reading_Using_CNN_and_LSTM/training/2nd Model checkpoint(ac- 78 and valacc - 63/model.h5'

def read_model(filepath,weights_filename='untrained_weight.h5',
               topo_filename='untrained_topo.json'):
    print("Reading Model from " + weights_filename + " and " + topo_filename)
    print("Please wait, it takes time.")
    model = load_model(path)
    #model.load_weights(weights_filename)
    print("Finish Reading!")
    rmsprop(lr=0.0001, rho=0.9, epsilon=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

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
    time = cap.get(cv2.CAP_PROP_POS_MSEC)

    # For each detected face, find the landmark.
    for (i, rect) in enumerate(rects):
        # Make the prediction and transfom it to numpy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        minn = np.asarray(shape[48:68])

        min2 = minn.min(0)
        max2 = minn.max(0)

        x2,y2=min2
        w2,h2 = max2

        cropped = image[y2:h2,x2:w2]


    cropped = cv2.resize(cropped,(weight,height))

    model = read_model(path)

    model.predict(cropped)

    print(len(cropped))

    cv2.imshow("Output", cropped)

    cv2.waitKey(1)


cv2.destroyAllWindows()
cap.release()