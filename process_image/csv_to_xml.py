import os
import xml.etree.ElementTree as et
import structure_csv as structureClass
import csv

with open('mouthData.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)

    print(csv_reader)
        # def structure(file, w, h, clas, xMins, yMins, xMax, yMax):
        # structureClass.structure(line[0], line[1], line[2], line[3], line[4], line[5], line[6], line[7])
        # print(line[0].replace('.png', ''))


print('hello world')
