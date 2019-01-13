import numpy as np
import os

def framesForWord(normalizedMouth,
                  framecount=1,
                  filepath=''):

   filepath = filepath.split('.', 1)[0] + ".align"
   align = open(filepath, 'r')

   data = align.readlines()

   for x in data:
      endframe = x.split()
      norEndframe = int(endframe[1])/1000
      word = endframe[2]

      if norEndframe >= framecount and word != "sil":

         image_min = normalizedMouth[normalizedMouth > 0].min()
         image_max = normalizedMouth[normalizedMouth > 0].max()
         normalizedMouth = (normalizedMouth - image_min) / (float(image_max - image_min))

         print(normalizedMouth)

         #SAVING TO THE FILE


      else:
         continue





def savedFile(speaker_input_train,
              speaker_input_test,
              speaker_output_train,
              speaker_output_test,
              filePath):

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

path = 'D:/RawDatasets/final_project/dataset/s1/bbaf2n.mpg'

data = framesForWord(filepath=path)

