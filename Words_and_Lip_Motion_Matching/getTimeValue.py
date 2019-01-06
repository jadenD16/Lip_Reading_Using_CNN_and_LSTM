from imutils import face_utils
import dlib
import cv2
import numpy as np

# p = our pre-treined model directory, on my case, it's on the same script's diretory.
p = "C:/Users/javinarfamily/PycharmProjects/Thesis/Lip_Reading_Using_CNN_and_LSTM/process_image/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

cap = cv2.VideoCapture('D:\\RawDatasets\\s1\\bbaf2n.mpg')

while True:
    # Getting out image by webcam
    _, image = cap.read()
    # Converting the image to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Get faces into webcam's image
    rects = detector(gray, 0)
    mask = np.zeros(image.shape, dtype="uint8")

    time = cap.get(cv2.CAP_PROP_POS_MSEC)

    print(time)

    # For each detected face, find the landmark.
    for (i, rect) in enumerate(rects):
        # Make the prediction and transfom it to numpy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        #x`xprint(shape[48:60])

        # Draw on our image, all the finded cordinate points (x,y)
        #for (x, y) in shape[48:60]:
        #cv2.fillPoly(mask,)
        cv2.polylines(image, [shape[48:67]], True, (0, 255, 255))
        #cv2.fillPoly(mask,[shape[48:60]],(0,255,255))


    # Show the image
    cv2.imshow("Output", image)

    cv2.waitKey(1)


cv2.destroyAllWindows()
cap.release()