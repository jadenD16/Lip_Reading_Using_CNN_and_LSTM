import numpy as np
import os
import cv2

wordId = np.load('C:\\Users\\Jaden\\PycharmProjects\\Thesis\\Lip_Reading_Using_CNN_and_LSTM\\Words_and_Lip_Motion_Matching\\wordIdx.npy')

def setAlignFile(filepath):

   filepath = filepath.split('.', 1)[0] + ".align"
   alignFile = open(filepath, 'r')
   alignFile = alignFile.readlines()

   return alignFile

def framesForWord(normalizedMouth,
                   framecount, alignFile):

   for align in alignFile:
      align = align.split()

      endframe = int(align[1])/1000
      startFrame = int(align[0])/1000
      word = align[2].split('/',1)
      word = word[0]

      if endframe >= framecount and startFrame <= frameCount:
         if word != "sil" and word != "sip":

            wordArray = normalizedMouth
            print(word + ' ' + str(frameCount)+' ')

            return word
            break

      else:
         continue

def savedFile():

   #DESTINATION PATH FOR THE ALIGNED WORD
   destination_path = 'D:/RawDatasets/final_project/dataset/lowquality_wordalignment/'

   speaker_input_train = []
   speaker_output_train = []
   speaker_input_test = []
   speaker_output_test = []


   folder = filePath.split('/', 5)[4]
   count = folder.split('s')
   count = count[1]
   print(count)

   #KULANG PA ITO NG CONDITIONS..
   speaker_input_train = np.asarray(speaker_input_train)
   speaker_input_test = np.asarray(speaker_input_test)

   if not os.path.exists(destination_path):
      os.makedirs(destination_path)
   f1 = open(destination_path + 'speaker_input_train' + str(count) + '.npz', "wb")
   np.savez_compressed(f1, speaker_input_train)

   f2 = open(destination_path + 'speaker_input_test' + str(count) + '.npz', "wb")
   np.savez_compressed(f2, speaker_input_test)

   f3 = open(destination_path + 'speaker_output_train' + str(count), "wb")
   np.save(f3, speaker_output_train)

   f4 = open(destination_path + 'speaker_output_test' + str(count), "wb")
   np.save(f4, speaker_output_test)

path = 'D:/Datasets/s1/bbaf2n.mpg'
alignFile = setAlignFile(path)
vid = cv2.VideoCapture(path)

frameCount=1

while(vid.isOpened()):

   r,frame = vid.read()

   data = framesForWord(frame, frameCount,alignFile)

   frameCount+=1

   cv2.imshow('mamaxzs', frame)
   cv2.waitKey(1)

vid.release()