import cv2
import numpy as np
import  glob


mouth_cascade = cv2.CascadeClassifier('C:\\Users\\Jaden\\Downloads\\haarcascade_mcs_mouth.xml')
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

        for (x,y,w,h) in mouth_rects:
            y = int(y - 0.15*h)

            print(str(x)+" "+str(y))

            crop = frame[y:y + (h+3), (x-5):x + (w+4)]
            resized = cv2.resize(crop, (w + 80, h + 50))

            if y != 0 and (evenPicker%2) != 0:
                pictCount += 1
                cv2.imwrite('D:\\Datasets\\picts\\pict' + str(pictCount) + '.png', resized)
                print(str(h) + str(w))
            break

        evenPicker += 1
        print('Saving pictures from: '+filename)
    cap.release()

print('Saving pictures done.....')
