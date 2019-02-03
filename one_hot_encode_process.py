import numpy as np
import os
import cv2

def setAlignFile(filepath):

   filepath = filepath.split('.', 1)[0] + ".align"
   alignFile = open(filepath, 'r')
   alignFile = alignFile.readlines()


   return alignFile

def normalizedMouth(mouth):

    image_min = mouth[mouth > 0].min()
    image_max = mouth[mouth > 0].max()

    mouth = (mouth - image_min) / (float(image_max - image_min))

    return mouth

previousWord = 'sil'
def framesForWord(normalizedMouth, frameCount, alignFile):

    isMatch = False
    isEmpty = False

    global previousWord
    lineNumber=0
    for align in alignFile:
      align = align.split()
      lineNumber += 1
      endframe = int(align[1])/1000
      startFrame = int(align[0])/1000

      #get the word
      word = align[2].split('/',1)
      word = word[0]

      if endframe >= frameCount and startFrame <= frameCount:
         if word != "sil" and word != "sip" and word != 'sp':

            if previousWord == word:
                isMatch = True
                isEmpty = True
                break

            else:
                previousWord = word
                isMatch = False
                isEmpty = True
                break

      else:
          if lineNumber >= len(alignFile):
              isMatch = False
              isEmpty = False

          continue

    return isMatch, isEmpty, previousWord