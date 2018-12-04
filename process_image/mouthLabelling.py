import cv2
import numpy as np
import  glob
from Lip_Reading_Using_CNN_and_LSTM.process_image import  mouthTracker as mt
import pandas as pd
import csv

mouth_cascade = cv2.CascadeClassifier('Lip_Reading_Using_CNN_and_LSTM\\process_image\\haarcascade_mcs_mouth.xml')
mouthCsv = 'Lip_Reading_Using_CNN_and_LSTM\\process_image\\mouthData.csv'
pictCount=0
evenPicker = 1

for filename in glob.glob('D:\\Datasets\\**\\*.mpg'):

    frame_counter = 0
    cap = cv2.VideoCapture(filename)
    ds_factor = 0.5

    while True:
        ret, frame = cap.read()

        frame_counter += 1

        frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if cap.get(cv2.CAP_PROP_FRAME_COUNT)==frame_counter:
            break

        mouth_rects = mouth_cascade.detectMultiScale(frame, 1.7, 11)
        mouthPoints = []

        for (x,y,w,h) in mouth_rects:
            y = int(y - 0.15*h)

            if (evenPicker%2) != 0:
                pictCount += 1
                # cv2.imwrite('‪F:\\Datasets\\newpicture\pic' + str(pictCount) + '.png', frame)

                #get the coordinates/points of interest
                mouthPoints = mt.getMouthPoints(gray)
                #get  the x and y mean
                ymax = y+h
                xmax = x+w
                xmin = x
                ymin = y
                pictname = 'pict' + str(pictCount) + '.png'

                with open('train_labels2.csv', 'r') as csv_file:
                    csv_reader = csv.reader(csv_file)

                    with open('train_labels2.csv', 'a',newline='') as csvappend_file:
                        csv_writer = csv.writer(csvappend_file)
                        csv_writer.writerow([pictname, '180', '144', 'Mouth', xmin, ymin, xmax, ymax])

                         #debugging tool
                        print(pictname)
                        print(xmin)
                        print(ymin)
                        print(xmax)
                        print(ymax)
                        # pd.read_csv(mouthCsv)
                        print('----------------------------------')

        evenPicker += 1
       # print('Saving pictures from: '+filename)

    cap.release()

#print('Saving pictures done.....')
