import pprint, pickle
import cv2
from time import sleep
import os

outputFolder = "../resource/data_new/zq/20/"

cv2.namedWindow("frame")
cv2.moveWindow("frame", 700, 400)

outputFileIndex = 1
maxLength = 0
while outputFileIndex < 30:
    image_pkl_file_path = outputFolder + "frame_" + str(outputFileIndex) + "_image.pkl"
    if os.path.isfile(image_pkl_file_path):
        print("file index: " + str(outputFileIndex))
        image_pkl_file = open(outputFolder + "frame_" + str(outputFileIndex) + "_image.pkl", 'rb')
        # image_pkl_file.seek(0)
        mouth_image_list = pickle.load(image_pkl_file)
        valid_len = min(len(mouth_image_list), 70)
        mouth_image_list = mouth_image_list[0:valid_len]
        image_pkl_file.close()

        maxLength = max(maxLength, len(mouth_image_list))
        print("len: {}    max-len: {}".format(len(mouth_image_list), maxLength))
        # print(len(mouth_lip_list))

        for i in range(0, len(mouth_image_list)):
            image = mouth_image_list[i]
            if i == len(mouth_image_list)-1:
                cv2.putText(image, 'done ' + str(outputFileIndex), (5, 66), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow('frame', image)
            key = cv2.waitKey(30) & 0xFF
            # sleep(0.05)
    
    key = cv2.waitKey(0) & 0xFF
    if key == ord('j'):
        outputFileIndex = max(1, outputFileIndex-1)
    elif key == ord('k'):
        outputFileIndex = min(30, outputFileIndex+1)
    elif key == ord('q'):
        print("invalid file: " + str(outputFileIndex))
        break

cv2.destroyAllWindows()
