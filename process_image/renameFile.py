import os

pictcount = 0
os.chdir('C:\\Tensorflow-Models\\models\\research\\object_detection\\annotations\\mask')
newFilename = 'pict'
for file in os.listdir():
    filename, file_ext = os.path.splitext(file)
    # print(filename)
    mask, num = filename.split('mask')


    new_name =  '{}{}{}'.format(newFilename,num,file_ext)
    os.rename(file, new_name)
