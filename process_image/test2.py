from imutils import face_utils
import dlib
import cv2
import numpy as np

# p = our pre-treined model directory, on my case, it's on the same script's diretory.
p = "C:\\Users\\Jaden\\Downloads\\shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

cap = cv2.VideoCapture('D:\\Datasets\\s1\\bbaf2n.mpg')
x2=0
y2=0
w2=0
h2=0
counter = 0

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

        #x`xprint(shape[48:60])

        # Draw on our image, all the finded cordinate points (x,y)
        #for (x, y) in shape[48:60]:
        #cv2.fillPoly(mask,)
        #cv2.polylines(image, [shape[48:67]], True, (0, 255, 255))
        #cv2.fillPoly(mask,[shape[48:60]],(0,255,255))

        if counter == 1:
            minn = np.asarray(shape[48:68])

            min2 = minn.min(0)
            max2 = minn.max(0)

            x2, y2 = min2
            w2, h2 = max2

            y2 -=7
            h2 +=4

        cropped = image[y2:h2,x2:w2]

        print(len(cropped))

        #cv2.rectangle(image, (y2,h2), (x2,w2), (255, 0, 0), thickness=3)
    # Show the image
    cv2.imshow("Output", cropped)

    cv2.waitKey(0)


cv2.destroyAllWindows()
cap.release()