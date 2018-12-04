from imutils import face_utils
import dlib

def getMouthPoints(frame):

    p = "C:\\Users\\Jaden\\PycharmProjects\\Thesis\\Lip_Reading_Using_CNN_and_LSTM\process_image\\shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)

    # Get faces into webcam's image
    rects = detector(frame, 0)

    # For each detected face, find the landmark.
    for (i, rect) in enumerate(rects):
        # Make the prediction and transfom it to numpy array
        shape = predictor(frame, rect)
        shape = face_utils.shape_to_np(shape)

    return shape[48:60]