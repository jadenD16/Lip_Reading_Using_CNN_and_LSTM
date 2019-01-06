import numpy
import os
from tempfile import TemporaryFile

source_video_path = 'D:/RawDatasets/'
source_wordalignment_path = 'D:/RawDatasets/align'
destination_path = 'D:/RawDatasets/wordAlign/'

file = TemporaryFile()
for cnt1 in range(1, 33):
    if cnt1 != 8 and cnt1 != 21:
        vidPath = source_video_path +'s'+str(cnt1)+'/'
        videoname_list = next(os.walk(vidPath))
        speaker_input_train =[]
        speaker_output_train = []
        speaker_input_test = []
        speaker_output_test = []
        word_dict = {}

        for video_name in videoname_list[2]:

            align_filename = video_name.split('.', 1)[0] + ".align"
            fileptr = open(source_wordalignment_path +'/s' + str(cnt1)+'/align/'+align_filename, "r+")
            print(source_wordalignment_path + '/s' + str(cnt1) + '/align/' + align_filename)
            sentence_framesdata = fileptr.read().splitlines()
            for word_framedata in sentence_framesdata:
                 test_flag = False
                 word_array = numpy.zeros((40,1600))
                 starting_frame = word_framedata.split()[0]
                 starting_frame = int(starting_frame) / 1000
                 starting_frame += 1
                 ending_frame = word_framedata.split()[1]
                 ending_frame = int(ending_frame) / 1000
                 word = word_framedata.split()[2]
                 if word not in word_dict:
                     test_flag = True
                     word_dict[word] = 1
                 elif word_dict[word] < 5:
                     word_dict[word] += 1
                     test_flag = True
                 else:
                     word_dict[word] += 1
                 word_index = 0

                 if word != "sil":
                    while starting_frame <= ending_frame:
                        print(starting_frame,ending_frame, word)
                        f = open(source_video_path + 's' + str(cnt1) + '/' + video_name + '/' + str(int(starting_frame)), "rb")
                        # f = open(source_video_path +'s' + str(cnt1)+'/'+video_name,"rb")
                        print(f)
                        image = numpy.load(f)
                        image = numpy.resize(image, 1600)
                        word_array[word_index] = image
                        word_index += 1
                        starting_frame += 1

                    if test_flag == False:
                        speaker_input_train.append(word_array)
                        speaker_output_train.append(word)
                    else:
                        speaker_input_test.append(word_array)
                        speaker_output_test.append(word)
        speaker_input_train = numpy.asarray(speaker_input_train)
        speaker_input_test = numpy.asarray(speaker_input_test)

    if not os.path.exists(destination_path):
        os.makedirs(destination_path)
    f1 =open(destination_path + 'speaker_input_train'+str(cnt1) + '.npz', "wb")
    numpy.savez_compressed(f1,speaker_input_train)

    f2 = open(destination_path + 'speaker_input_test'+str(cnt1) + '.npz', "wb")
    numpy.savez_compressed(f2, speaker_input_test)

    f3 = open(destination_path + 'speaker_output_train'+str(cnt1), "wb")
    numpy.save(f3, speaker_output_train)

    f4 = open(destination_path + 'speaker_output_test'+str(cnt1), "wb")
    numpy.save(f4, speaker_output_test)