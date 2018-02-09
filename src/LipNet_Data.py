import pprint, pickle
import os
import numpy as np
import cv2
from random import shuffle
import sys

def get_array_data():
    print('get_array_data() start')
    subjects = ['gyz', 'hpk', 'hy', 'hzr', 'lj', 'll', 'lyq', 'mq',
                'plh', 'pxy', 'sk', 'swn', 'wn', 'wrl', 'wxy', 'yx',
                'yyk', 'yzc', 'zjw', 'zmy', 'zq', 'ztx']
    for subject in subjects:
        print(subject)
        subjectFolder = "../resource/data_new/" + subject + "/"
        subjectOutputFolder = "../resource/data_new_augmented_array/" + subject + "/"
        if not os.path.exists(subjectOutputFolder):
            os.mkdir(subjectOutputFolder)
        subjectFlipOutputFolder = "../resource/data_new_augmented_array/" + subject + "_flip/"
        if not os.path.exists(subjectFlipOutputFolder):
            os.mkdir(subjectFlipOutputFolder)

        for gestureId in range(1, 43+1):
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
            for fileIndex in range(1, 30):
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


def check_array_and_augmented_data(subject = 'wxy', gestureId = 6, fileIndex = 3):
    image_pkl_output_file_path = "../resource/data_new_augmented_array/" + subject + "/" + str(gestureId) + "/" + "frame_" + str(fileIndex) + "_image.pkl"
    image_pkl_flip_output_file_path = "../resource/data_new_augmented_array/" + subject + "_flip/" + str(gestureId) + "/" + "frame_" + str(fileIndex) + "_image.pkl"
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
            cv2.waitKey(30)

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
            # mouth_image_array = pickle.load(image_pkl_file).astype(dtype=np.float32)  # frame_n, img_h, img_w, img_c
            mouth_image_array = pickle.load(image_pkl_file)
            image_pkl_file.close()
        except ValueError:
            print(ID)

    print("check test done")

def check_array_and_augmented_data_all_one_out(out_name = 'wxy', group = 1):
    outputFolder = '../resource/new_training_testing_list/group_' + str(group) + '/'
    filename = outputFolder + 'training_list_' + out_name + '_out.txt'
    with open(filename) as f:
        list_IDs = f.readlines()
        list_IDs = [x.strip() for x in list_IDs]

    for ID in list_IDs:
        subject, gestureId, filename = ID.split("-")
        file_path = "../resource/data_new_augmented_array/" + subject + "/" + str(gestureId) + "/" + filename
        try:
            image_pkl_file = open(file_path, 'rb')
            mouth_image_array = pickle.load(image_pkl_file).astype(dtype=np.float32)  # frame_n, img_h, img_w, img_c
            image_pkl_file.close()
        except ValueError:
            print(ID)

    print("check train done")

    filename = outputFolder + 'testing_list_' + out_name + '_out.txt'
    with open(filename) as f:
        list_IDs = f.readlines()
        list_IDs = [x.strip() for x in list_IDs]

    for ID in list_IDs:
        subject, gestureId, filename = ID.split("-")
        file_path = "../resource/data_new_augmented_array/" + subject + "/" + str(gestureId) + "/" + filename
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

def split_to_training_testing_set_one_out(out_name = 'wxy', group = 1, gestureIDs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]):
    print('** {:40s}'.format('split_to_training_testing_set() start') )
    subjects = ['gyz', 'hpk', 'hy', 'hzr', 'lj', 'll', 'lyq', 'mq',
                'plh', 'pxy', 'sk', 'swn', 'wn', 'wrl', 'wxy', 'yx',
                'yyk', 'yzc', 'zjw', 'zmy', 'zq', 'ztx']
    subjects_augmented = subjects + [subject + '_flip' for subject in subjects]
    trainingIDs = []
    testingIDs = []
    for subject in subjects_augmented:
        # print(subject)
        for gestureId in gestureIDs:
            subject_gesture_dir = "../resource/data_new_augmented_array/" + subject + "/" + str(gestureId) + "/"
            subject_gesture_filenames = []
            for file in os.listdir(subject_gesture_dir):
                if file.endswith(".pkl"):
                    subject_gesture_filenames.append(file)
                    ID = subject + '-' + str(gestureId) + '-' + file
                    if subject == out_name or subject == out_name+'_flip':
                        testingIDs.append(ID)
                    else:
                        trainingIDs.append(ID)

    outputFolder = '../resource/new_training_testing_list/group_' + str(group) + '/'
    if not os.path.exists(outputFolder):
        os.mkdir(outputFolder)

    file = open(outputFolder + 'training_list_' + out_name + '_out.txt', 'w')
    print(file.name)
    for ID in trainingIDs:
        file.write(ID + "\n")
    file.close()
    file = open(outputFolder + 'testing_list_' + out_name + '_out.txt', 'w')
    print(file.name)
    for ID in testingIDs:
        file.write(ID + "\n")
    file.close()
    print("### traing size: {}   testing size: {}".format(len(trainingIDs), len(testingIDs)))
    print('** {:40s}'.format('split_to_training_testing_set() end'))
    print('-----------------------------------------')

if __name__ == "__main__":
    # check_array_and_augmented_data(subject='zmy', gestureId=1, fileIndex=11)

    outman = ''
    gestureIDs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    group = 8  # 1: open apps  2: preferences  3: wechat  4: edit text  5: notification

    if len(sys.argv) == 3:
        print(str(sys.argv))
        group = int(sys.argv[1])
        outman = sys.argv[2]

    if group == 1:
        gestureIDs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    elif group == 2:
        gestureIDs = [11, 12, 13, 14, 15, 16, 17, 42, 43]
    elif group == 3:
        gestureIDs = [19, 20, 22, 23, 24, 25, 26]
    elif group == 4 :
        gestureIDs = [27, 28, 29, 30, 31, 33, 34, 35]
    elif group == 5:
        gestureIDs = [36, 37, 38, 39, 40, 41]
    elif group == 6:
        gestureIDs = [11, 12, 13, 14, 15, 16, 17, 42, 43, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    elif group == 7:
        gestureIDs = [11, 12, 13, 14, 15, 16, 17, 42, 43, 19, 20, 22, 23, 24, 25, 26]
    elif group == 8:
        gestureIDs = [11, 12, 13, 14, 15, 16, 17, 42, 43, 27, 28, 29, 30, 31, 33, 34, 35]
    elif group == 9:
        gestureIDs = [11, 12, 13, 14, 15, 16, 17, 42, 43, 36, 37]

    split_to_training_testing_set_one_out(out_name=outman, group=group, gestureIDs=gestureIDs)
    # check_array_and_augmented_data_all_one_out(out_name=outman, group=group)




