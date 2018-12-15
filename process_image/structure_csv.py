from xml.etree.ElementTree import ElementTree
from xml.etree.ElementTree import Element
import xml.etree.ElementTree as et


def structure(file, w, h, clas, xMins, yMins, xMax, yMax):
    root = Element('annotation')
    folder = Element('folder')
    filename = Element('filename')
    path = Element('path')
    source = Element('source')
    database = Element('database')
    size = Element('size')
    width = Element('width')
    height = Element('height')
    Depth = Element('depth')
    segmented = Element('segmented')
    object = Element('object')
    name = Element('name')
    pose = Element('pose')
    truncated = Element('truncated')
    difficult = Element('difficult')
    boundbox = Element('bndbox')
    xmin = Element('xmin')
    ymin = Element('ymin')
    xmax = Element('xmax')
    ymax = Element('ymax')
    tree = ElementTree(root)

    # appending the xml file
    root.append(folder)
    root.append(filename)
    root.append(path)
    root.append(source)
    source.append(database)
    root.append(size)
    size.append(width)
    size.append(height)
    size.append(Depth)
    root.append(segmented)
    root.append(object)
    object.append(name)
    object.append(pose)
    object.append(truncated)
    object.append(difficult)
    object.append(boundbox)
    boundbox.append(xmin)
    boundbox.append(ymin)
    boundbox.append(xmax)
    boundbox.append(ymax)

    # data for each file
    folder.text ='mask'
    path.text = 'C:\\Tensorflow-Models\\models\\research\\object_detection\\annotations\\mask\\' + file
    Depth.text = '3'
    segmented.text = '0'
    difficult.text = '0'
    truncated.text = '0'
    database.text = 'Unknown'
    pose.text = 'Unspecified'
    width.text = w
    height.text = h
    filename.text = file
    name.text = clas
    xmin.text = xMins
    ymin.text = yMins
    xmax.text = xMax
    ymax.text = yMax

    print(et.tostring(root))
    tree.write(open(file.replace('.png','') + '.xml', 'wb'))
