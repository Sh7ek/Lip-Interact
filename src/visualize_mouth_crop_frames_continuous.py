import pprint, pickle
import cv2
from time import sleep
import os

outputFolder = "../resource/data/sk/12/"

cv2.namedWindow("frame");
cv2.moveWindow("frame", 1000, 400);

outputFileIndex = 1
maxLength = 0
while outputFileIndex < 25:
    image_pkl_file_path = outputFolder + "frame_" + str(outputFileIndex) + "_image.pkl"
    if os.path.isfile(image_pkl_file_path):
        print("file index: " + str(outputFileIndex))
        image_pkl_file = open(outputFolder + "frame_" + str(outputFileIndex) + "_image.pkl", 'rb')
        image_pkl_file.seek(0)
        mouth_image_list = pickle.load(image_pkl_file)
        valid_len = min(len(mouth_image_list), 70)
        mouth_image_list = mouth_image_list[0:valid_len]
        image_pkl_file.close()

        lip_pkl_file = open(outputFolder + "frame_" + str(outputFileIndex) + "_lip.pkl", 'rb')
        mouth_lip_list = pickle.load(lip_pkl_file)
        lip_pkl_file.close()

        maxLength = max(maxLength, len(mouth_image_list))
        print("len: {}    max-len: {}".format(len(mouth_image_list), maxLength))
        # print(len(mouth_lip_list))

        for i in range(0, len(mouth_image_list)):
            image = mouth_image_list[i]
            lip = mouth_lip_list[i]
            for (x, y) in lip:
                cv2.circle(image, (int(x), int(y)), 1, (255, 255, 255), -1)
            cv2.imshow('frame', image)
            key = cv2.waitKey(50) & 0xFF
            # sleep(0.05)
    
    key = cv2.waitKey(0) & 0xFF
    if key == ord('j'):
        outputFileIndex = max(1, outputFileIndex-1)
    elif key == ord('k'):
        outputFileIndex = min(24, outputFileIndex+1)
    elif key == ord('q'):
        print("invalid file: " + str(outputFileIndex))
        break

cv2.destroyAllWindows()
