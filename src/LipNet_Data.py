import pprint, pickle
import os
import numpy as np
from keras.preprocessing import sequence
import cv2

outputFolder = "../resource/data/sk/12/"

maxLength = 0
minLength = 1000

data_list = []
for outputFileIndex in range(1, 25):
    image_pkl_file_path = outputFolder + "frame_" + str(outputFileIndex) + "_image.pkl"
    if os.path.isfile(image_pkl_file_path):
        # print("file index: " + str(outputFileIndex))
        image_pkl_file = open(outputFolder + "frame_" + str(outputFileIndex) + "_image.pkl", 'rb')
        mouth_image_list = pickle.load(image_pkl_file)
        # mouth_image_array = np.asarray(mouth_image_list, dtype=np.float32)
        mouth_image_array = np.asarray(mouth_image_list)
        maxLength = max(maxLength, mouth_image_array.shape[0])
        minLength = min(minLength, mouth_image_array.shape[0])
        data_list.append(mouth_image_array)

data_array = np.asarray(data_list)

print('min_len {}    max_len {}'.format(minLength, maxLength))
print(data_array.shape[0])
print(type(data_array))


check_i = 20
check_frame_in_image = 17
check_h = 40
check_w = 50
for i in range(0, data_array.shape[0]):
    mouth_image_array = data_array[i]
    print("{}  {}  {}  {}".format(i, mouth_image_array.shape, type(mouth_image_array), mouth_image_array.dtype))

print(data_array[check_i][check_frame_in_image][check_h][check_w])
cv2.namedWindow("frame1");
cv2.moveWindow("frame1", 800, 400);
cv2.imshow('frame1', data_array[check_i][check_frame_in_image])

data_array = sequence.pad_sequences(data_array, maxlen=30, padding='post', truncating= 'post', dtype=np.uint8)

print("")
for i in range(0, data_array.shape[0]):
    mouth_image_array = data_array[i]
    print("{}  {}  {}  {}".format(i, mouth_image_array.shape, type(mouth_image_array), mouth_image_array.dtype))

print(data_array[check_i][check_frame_in_image][check_h][check_w])
cv2.namedWindow("frame2");
cv2.moveWindow("frame2", 1300, 400);
cv2.imshow('frame2', data_array[check_i][check_frame_in_image])

cv2.waitKey(0)
cv2.destroyAllWindows()
