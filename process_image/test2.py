import process_GRIDcorpus as pG
import cv2 as cv
import  glob

for filename in glob.glob('D:\\Datasets\\s1\\*.mpg'):

    frame_counter = 0
    cap = cv.VideoCapture(filename)
    ds_factor = 0.5

    while True:
        ret, frame = cap.read()

        frame_counter+=1
        if cap.get(cv.CAP_PROP_FRAME_COUNT)==frame_counter:
            break

        face = pG.findFaceRect(frame)

        mouth=pG.findMouthMeanInFaceRect(frame)

        print(mouth)
        print(filename)

