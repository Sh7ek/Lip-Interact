import pprint, pickle
import os
import numpy as np
from keras.preprocessing import sequence
import cv2
from shutil import copyfile
from random import shuffle

def get_array_data():
    print('get_array_data() start')
    subjects = ['gyz', 'hpk', 'lyq', 'plh', 'sk', 'swn', 'wrl', 'xwj', 'yzc', 'zq', 'ztx']
    for subject in subjects:
        print(subject)
        subjectFolder = "../resource/data/" + subject + "/"
        subjectOutputFolder = "../resource/data_augmented_array/" + subject + "/"
        if not os.path.exists(subjectOutputFolder):
            os.mkdir(subjectOutputFolder)
        subjectFlipOutputFolder = "../resource/data_augmented_array/" + subject + "_flip/"
        if not os.path.exists(subjectFlipOutputFolder):
            os.mkdir(subjectFlipOutputFolder)

        for gestureId in range(1, 12+1):
            # print("{} {}".format(subject, gestureId))
            subjectGestureFolder = subjectFolder + str(gestureId) + "/"
            subjectGestureOutputFolder = subjectOutputFolder + str(gestureId) + "/"
            if not os.path.exists(subjectGestureOutputFolder):
                os.mkdir(subjectGestureOutputFolder)
            subjectGestureFlipOutputFolder = subjectFlipOutputFolder + str(gestureId) + "/"
            if not os.path.exists(subjectGestureFlipOutputFolder):
                os.mkdir(subjectGestureFlipOutputFolder)

            if not os.path.isdir(subjectGestureFolder):
                print("wrong path: " + subjectGestureFolder)
                exit(100)
            for fileIndex in range(1, 25):
                image_pkl_file_path = subjectGestureFolder + "frame_" + str(fileIndex) + "_image.pkl"
                if os.path.isfile(image_pkl_file_path):
                    image_pkl_file = open(image_pkl_file_path, 'rb')
                    mouth_image_list = pickle.load(image_pkl_file)
                    image_pkl_file.close()
                    mouth_image_array = np.asarray(mouth_image_list)

                    image_pkl_output_file_path = subjectGestureOutputFolder + "frame_" + str(fileIndex) + "_image.pkl"
                    output = open(image_pkl_output_file_path, 'wb')
                    pickle.dump(mouth_image_array, output)
                    output.close()

                    for fr in range(0, mouth_image_array.shape[0]):
                        mouth_image_array[fr] = cv2.flip(mouth_image_array[fr], 1)
                    image_pkl_flip_output_file_path = subjectGestureFlipOutputFolder + "frame_" + str(fileIndex) + "_image.pkl"
                    output = open(image_pkl_flip_output_file_path, 'wb')
                    pickle.dump(mouth_image_array, output)
                    output.close()
    print('get_array_data() end')


def check_array_and_augmented_data():
    subjects = ['gyz', 'hpk', 'lyq', 'plh', 'sk', 'swn', 'wrl', 'xwj', 'yzc', 'zq', 'ztx']
    subject = "ztx"
    gestureId = 6
    fileIndex = 3
    image_pkl_output_file_path = "../resource/data_augmented_array/" + subject + "/" + str(gestureId) + "/" + "frame_" + str(fileIndex) + "_image.pkl"
    image_pkl_flip_output_file_path = "../resource/data_augmented_array/" + subject + "_flip/" + str(gestureId) + "/" + "frame_" + str(fileIndex) + "_image.pkl"
    if os.path.isfile(image_pkl_output_file_path):
        image_pkl_file = open(image_pkl_output_file_path, 'rb')
        mouth_image_array = pickle.load(image_pkl_file)
        image_pkl_file.close()

        image_flip_pkl_file = open(image_pkl_flip_output_file_path, 'rb')
        mouth_image_flip_array = pickle.load(image_flip_pkl_file)
        image_flip_pkl_file.close()

        cv2.namedWindow("frame1");
        cv2.moveWindow("frame1", 800, 400);
        cv2.namedWindow("frame2");
        cv2.moveWindow("frame2", 910, 400);

        for i in range(0, mouth_image_array.shape[0]):
            image = mouth_image_array[i]
            image_flip = mouth_image_flip_array[i]
            cv2.imshow('frame1', image)
            cv2.imshow('frame2', image_flip)
            cv2.waitKey(40)

        key = cv2.waitKey(0)
        cv2.destroyAllWindows()

def check_array_and_augmented_data_all():
    # create batch generator
    filename = "../resource/training_list.txt"
    with open(filename) as f:
        list_IDs = f.readlines()
        list_IDs = [x.strip() for x in list_IDs]

    for ID in list_IDs:
        subject, gestureId, filename = ID.split("-")
        file_path = "../resource/data_augmented_array/" + subject + "/" + str(gestureId) + "/" + filename
        try:
            image_pkl_file = open(file_path, 'rb')
            mouth_image_array = pickle.load(image_pkl_file).astype(dtype=np.float32)  # frame_n, img_h, img_w, img_c
            image_pkl_file.close()
        except ValueError:
            print(ID)

    print("check train done")

    filename = "../resource/testing_list.txt"
    with open(filename) as f:
        list_IDs = f.readlines()
        list_IDs = [x.strip() for x in list_IDs]

    for ID in list_IDs:
        subject, gestureId, filename = ID.split("-")
        file_path = "../resource/data_augmented_array/" + subject + "/" + str(gestureId) + "/" + filename
        try:
            image_pkl_file = open(file_path, 'rb')
            mouth_image_array = pickle.load(image_pkl_file).astype(dtype=np.float32)  # frame_n, img_h, img_w, img_c
            image_pkl_file.close()
        except ValueError:
            print(ID)

    print("check test done")

def check_array_and_augmented_data_all_one_out(out_name = 'sk'):
    # create batch generator
    filename = "../resource/training_list_" + out_name + "_out.txt"
    with open(filename) as f:
        list_IDs = f.readlines()
        list_IDs = [x.strip() for x in list_IDs]

    for ID in list_IDs:
        subject, gestureId, filename = ID.split("-")
        file_path = "../resource/data_augmented_array/" + subject + "/" + str(gestureId) + "/" + filename
        try:
            image_pkl_file = open(file_path, 'rb')
            mouth_image_array = pickle.load(image_pkl_file).astype(dtype=np.float32)  # frame_n, img_h, img_w, img_c
            image_pkl_file.close()
        except ValueError:
            print(ID)

    print("check train done")

    filename = "../resource/testing_list_" + out_name + "_out.txt"
    with open(filename) as f:
        list_IDs = f.readlines()
        list_IDs = [x.strip() for x in list_IDs]

    for ID in list_IDs:
        subject, gestureId, filename = ID.split("-")
        file_path = "../resource/data_augmented_array/" + subject + "/" + str(gestureId) + "/" + filename
        try:
            image_pkl_file = open(file_path, 'rb')
            mouth_image_array = pickle.load(image_pkl_file).astype(dtype=np.float32)  # frame_n, img_h, img_w, img_c
            image_pkl_file.close()
        except ValueError:
            print(ID)

    print("check test done")


def split_to_training_testing_set(split = 0.8):
    print('** {:40s}'.format('split_to_training_testing_set() start') )
    subjects = ['gyz', 'hpk', 'lyq', 'plh', 'sk', 'swn', 'wrl', 'xwj', 'yzc', 'zq', 'ztx']
    subjects_augmented = subjects + [subject + '_flip' for subject in subjects]
    trainingIDs = []
    testingIDs = []
    for subject in subjects_augmented:
        # print(subject)
        for gestureId in range(1, 12 + 1):
            if gestureId == 6:
                continue
            subject_gesture_dir = "../resource/data_augmented_array/" + subject + "/" + str(gestureId) + "/"
            subject_gesture_filenames = []
            for file in os.listdir(subject_gesture_dir):
                if file.endswith(".pkl"):
                    subject_gesture_filenames.append(file)
                    shuffle(subject_gesture_filenames)
            cut = int(len(subject_gesture_filenames)*split)
            for i in range(0, cut):
                ID = subject + '-' + str(gestureId) + '-' + subject_gesture_filenames[i]
                trainingIDs.append(ID)
            for i in range(cut, len(subject_gesture_filenames)):
                ID = subject + '-' + str(gestureId) + '-' + subject_gesture_filenames[i]
                testingIDs.append(ID)

    file = open("../resource/training_list.txt", 'w')
    for ID in trainingIDs:
        file.write(ID + "\n")
    file.close()
    file = open("../resource/testing_list.txt", 'w')
    for ID in testingIDs:
        file.write(ID + "\n")
    file.close()
    print("### traing size: {}   testing size: {}".format(len(trainingIDs), len(testingIDs)))
    print('** {:40s}'.format('split_to_training_testing_set() end'))
    print('-----------------------------------------')

def split_to_training_testing_set_one_out(out_name = 'sk'):
    print('** {:40s}'.format('split_to_training_testing_set() start') )
    subjects = ['gyz', 'hpk', 'lyq', 'plh', 'sk', 'swn', 'wrl', 'xwj', 'yzc', 'zq', 'ztx']
    subjects_augmented = subjects + [subject + '_flip' for subject in subjects]
    trainingIDs = []
    testingIDs = []
    for subject in subjects_augmented:
        # print(subject)
        for gestureId in range(1, 12 + 1):
            if gestureId == 6:
                continue
            subject_gesture_dir = "../resource/data_augmented_array/" + subject + "/" + str(gestureId) + "/"
            subject_gesture_filenames = []
            for file in os.listdir(subject_gesture_dir):
                if file.endswith(".pkl"):
                    subject_gesture_filenames.append(file)
                    ID = subject + '-' + str(gestureId) + '-' + file
                    if subject == out_name or subject == out_name+'_flip':
                        testingIDs.append(ID)
                    else:
                        trainingIDs.append(ID)

    file = open("../resource/training_list_" + out_name + "_out.txt", 'w')
    for ID in trainingIDs:
        file.write(ID + "\n")
    file.close()
    file = open("../resource/testing_list_" + out_name + "_out.txt", 'w')
    for ID in testingIDs:
        file.write(ID + "\n")
    file.close()
    print("### traing size: {}   testing size: {}".format(len(trainingIDs), len(testingIDs)))
    print('** {:40s}'.format('split_to_training_testing_set() end'))
    print('-----------------------------------------')

if __name__ == "__main__":
    outman = 'ztx'
    split_to_training_testing_set_one_out(out_name=outman)
    check_array_and_augmented_data_all_one_out(out_name=outman)
