from imutils import face_utils
import dlib
import cv2
import numpy as np
from Lip_Reading_Using_CNN_and_LSTM import one_hot_encode_process as op
import  glob

# p = our pre-treined model directory, on my case, it's on the same script's diretory.
p = "C:\\Users\\Jaden\\Downloads\\shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

sentenceList = list()
for path in glob.glob('D:\\Datasets\\s1\\*.mpg'):

    cap = cv2.VideoCapture(path)
    x2 = 0
    y2 = 0
    w2 = 0
    h2 = 0
    counter = 0

    alignFile = op.setAlignFile(path)
    wordSequence = np.zeros((40, 1600))
    wCounter = 0

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

            #x`xprint(shape[48:60])

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
            normMouth = op.normalizedMouth(croppedMouth)

            normMouth = np.array(normMouth[:, 0:, :2])

            normMouth = normMouth.transpose(2, 0, 1).reshape(40,-1)

            normMouth = np.resize(normMouth, 1600)
            isMatch, isEmpty, previousWord = op.framesForWord(normMouth, counter, alignFile)

            if isMatch == True and isEmpty == True:
                wordSequence[wCounter] = normMouth
                wCounter += 1

            elif isMatch == False and isEmpty == True:

                if previousWord != "sil" and previousWord != "sip" and previousWord != 'sp':
                    print(previousWord)
                    wCounter = 0
                    sentenceList.append(wordSequence)
                    wordSequence = np.zeros((40,1600))
                    wordSequence[wCounter] = normMouth

        cv2.waitKey(1)
    cap.release()

    print('Done Saving from ' + path)
    print('Current List Shape: ' + str(np.shape(sentenceList)))
    np.save('C:\\Users\\Jaden\\PycharmProjects\\Thesis\\Lip_Reading_Using_CNN_and_LSTM\\sample',sentenceList)

