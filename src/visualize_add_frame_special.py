import pprint, pickle
import cv2
from time import sleep
import os

for gestureID in range(15, 15+1):
    outputFolder = "../resource/data_new/zq_origin/" + str(gestureID) + '/'
    targetFolder = "../resource/data_new/zq/" + str(gestureID) + '/'

    if not os.path.exists(targetFolder):
        os.mkdir(targetFolder)


    for outputFileIndex in range(0, 30):
        image_pkl_file_path = outputFolder + "frame_" + str(outputFileIndex) + "_image.pkl"
        if os.path.isfile(image_pkl_file_path):
            print("file index: " + str(outputFileIndex))
            image_pkl_file = open(outputFolder + "frame_" + str(outputFileIndex) + "_image.pkl", 'rb')
            mouth_image_list = pickle.load(image_pkl_file)
            valid_len = min(len(mouth_image_list), 70)
            mouth_image_list = mouth_image_list[0:valid_len]
            image_pkl_file.close()

            if len(mouth_image_list) < 35:
                mouth_image_list_new = []
                for i in range(0, len(mouth_image_list)):
                    mouth_image_list_new.append(mouth_image_list[i])
                    mouth_image_list_new.append(mouth_image_list[i])
                    image_pkl_output_file_path = targetFolder + "frame_" + str(outputFileIndex) + "_image.pkl"
                    output = open(image_pkl_output_file_path, 'wb')
                    pickle.dump(mouth_image_list_new, output)
                    output.close()
