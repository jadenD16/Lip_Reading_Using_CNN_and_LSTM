import numpy as np
import os
import cv2
from collections import defaultdict

#wordSequenceArray = np.zeros(40,1600)

previousWord = ''
frameperWord = list()
wordDict = {}
wordCount = 0

def setAlignFile(filepath):

   filepath = filepath.split('.', 1)[0] + ".align"
   alignFile = open(filepath, 'r')
   alignFile = alignFile.readlines()

   return alignFile

def framesForWord(normalizedMouth, framecount, alignFile):
   global previousWord, wordCount, frameperWord, wordDict

   for align in alignFile:
      align = align.split()

      endframe = int(align[1])/1000
      startFrame = int(align[0])/1000
      word = align[2].split('/',1)
      word = word[0]

      if endframe >= framecount and startFrame <= frameCount:
         if word != "sil" and word != "sip":

            wordArray = normalizedMouth

            if previousWord != word:
                previousWord = word
                wordCount = 0
                wordDict[word] = frameperWord
                frameperWord.clear()
                frameperWord.append(frame)

            else:
               wordCount += 1
               frameperWord.append(frame)
               break
      else:
         continue
   print(wordDict.keys)

path = 'D:/Datasets/s1/bbaf2n.mpg'
alignFile = setAlignFile(path)
vid = cv2.VideoCapture(path)
frameCount=0

while(vid.isOpened()):

    r,frame = vid.read()
    print(np.shape(frame))
    data = framesForWord(frame, frameCount,alignFile)
    frameCount+=1


    if cv2.CAP_PROP_FRAME_COUNT < frameCount:
        break

    cv2.waitKey(1)

    print(wordDict.keys())
    print(frameCount)
vid.release()