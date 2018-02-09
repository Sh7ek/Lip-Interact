from keras.models import load_model
from datetime import datetime
from keras.callbacks import CSVLogger, Callback
from keras import backend as K
from keras.optimizers import Adam
import os
import sys
import random
import numpy as np
import pickle

if __name__ == '__main__':

    group = 2
    outman = 'hpk'
    progress = 1

    if len(sys.argv) == 4:
        print(str(sys.argv))
        group = int(sys.argv[1])
        outman = sys.argv[2]
        progress = int(sys.argv[3])

    model_file_dir = '../resource/new_model/group_' + str(group) + '/'
    pre_testing_file_name = '../resource/new_training_testing_list/group_' + str(group) + '/testing_list_'+outman + '_out.txt'
    pre_testing_list = []

    gestureIDs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    if group == 1:
        gestureIDs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    elif group == 2:
        gestureIDs = [11, 12, 13, 14, 15, 16, 17, 42, 43]
    elif group == 3:
        gestureIDs = [19, 20, 22, 23, 24, 25, 26]
    elif group == 4:
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

    with open(pre_testing_file_name) as f:
        pre_testing_list = f.readlines()
        pre_testing_list = [x.strip() for x in pre_testing_list]

    new_training_list = []
    new_testing_list = []

    lastGestureID = -1
    currentFileList = []
    for line in pre_testing_list:
        if line.startswith(outman+'_flip'):
            break
        currentGestureID = int(line.split('-')[1])
        if currentGestureID != lastGestureID:
            if lastGestureID > 0:
                sample_file_names = []
                for i, item in enumerate(currentFileList):
                    if i < progress:
                        sample_file_names.append(item)
                    else:
                        s = int(random.random() * (i+1))
                        if s < progress:
                            sample_file_names[s] = item

                for item in sample_file_names:
                    new_training_list.append(item)
                for item in currentFileList:
                    if item not in sample_file_names:
                        new_testing_list.append(item)
            currentFileList = []
            currentFileList.append(line)
            lastGestureID = currentGestureID
        else:
            currentFileList.append(line)

    if len(currentFileList) > 0:
        sample_file_names = []
        for i, item in enumerate(currentFileList):
            if i < progress:
                sample_file_names.append(item)
            else:
                s = int(random.random() * (i + 1))
                if s < progress:
                    sample_file_names[s] = item

        for item in sample_file_names:
            new_training_list.append(item)
        for item in currentFileList:
            if item not in sample_file_names:
                new_testing_list.append(item)

    new_training_list_flip = []
    new_testing_list_flip = []
    for line in pre_testing_list:
        if line.startswith(outman+'_flip'):
            flip_gesture_id = int(line.split('-')[1])
            flip_file_label = int(line.split('_')[2])
            found = False
            for item in new_training_list:
                gesture_id = int(item.split('-')[1])
                file_label = int(item.split('_')[1])
                if gesture_id == flip_gesture_id and file_label == flip_file_label:
                    found = True
                    break
            if found:
                new_training_list_flip.append(line)
            else:
                new_testing_list_flip.append(line)

    new_training_list = new_training_list + new_training_list_flip
    new_testing_list = new_testing_list + new_testing_list_flip

    print("{} {} {}".format(len(pre_testing_list), len(new_training_list), len(new_testing_list)))

    X = np.zeros((len(new_training_list), 70, 80, 100, 3), dtype=np.float32)
    y = np.zeros((len(new_training_list)), dtype=int)

    # Generate data
    for i, ID in enumerate(new_training_list):
        # ID, aka file name format : "sk_flip-7-frame_11_image.pkl"
        subject, gestureId, filename = ID.split("-")

        file_path = "../resource/data_new_augmented_array/" + subject + "/" + str(gestureId) + "/" + filename
        image_pkl_file = open(file_path, 'rb')
        mouth_image_array = pickle.load(image_pkl_file).astype(dtype=np.float32)  # frame_n, img_h, img_w, img_c
        image_pkl_file.close()

        frame_n_temp = min(mouth_image_array.shape[0], 70)

        # normailize image by (X-X_Mean)/X_Std
        r_mean = np.mean(mouth_image_array[0:frame_n_temp, :, :, 0])
        g_mean = np.mean(mouth_image_array[0:frame_n_temp, :, :, 1])
        b_mean = np.mean(mouth_image_array[0:frame_n_temp, :, :, 2])

        r_std = np.std(mouth_image_array[0:frame_n_temp, :, :, 0])
        g_std = np.std(mouth_image_array[0:frame_n_temp, :, :, 1])
        b_std = np.std(mouth_image_array[0:frame_n_temp, :, :, 2])

        mouth_image_array[0:frame_n_temp, :, :, 0] = (mouth_image_array[0:frame_n_temp, :, :, 0] - r_mean) / r_std
        mouth_image_array[0:frame_n_temp, :, :, 1] = (mouth_image_array[0:frame_n_temp, :, :, 1] - g_mean) / g_std
        mouth_image_array[0:frame_n_temp, :, :, 2] = (mouth_image_array[0:frame_n_temp, :, :, 2] - b_mean) / b_std

        X[i, 0:frame_n_temp, :, :, :] = mouth_image_array[0:frame_n_temp, :, :, :]
        # X[i, frame_n_temp:self.frames_n] = mouth_image_array[mouth_image_array.shape[0]-1]  # padding the training instance with the last valid frame

        y[i] = gestureIDs.index(int(gestureId))  # start from 0
        Y = np.array([[1 if y[i] == j else 0 for j in range(len(gestureIDs))] for i in range(y.shape[0])])



    model_file_name = ''

    for file in os.listdir(model_file_dir):
        if file.startswith(outman) and file.endswith('.h5'):
            model_file_name = file
            break

    if len(model_file_name) == 0:
        exit(123)

    model_file_name = model_file_dir + model_file_name
    pre_model = load_model(model_file_name)

    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    pre_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    pre_model.fit(X, Y, batch_size=len(gestureIDs)*2, epochs=30, shuffle=False)


    # lipnet.model.save(outputFolder + outman + '_out_' + now + '_model.h5')

    # test
    total_instances = 0
    right_instances = 0
    for ID in new_testing_list:
        subject, gestureId, filename = ID.split("-")
        file_path = "../resource/data_new_augmented_array/" + subject + "/" + str(gestureId) + "/" + filename
        image_pkl_file = open(file_path, 'rb')
        mouth_image_array = pickle.load(image_pkl_file).astype(dtype=np.float32)  # frame_n, img_h, img_w, img_c
        image_pkl_file.close()

        frame_n_temp = min(mouth_image_array.shape[0], 70)
        X = np.zeros((1, 70, 80, 100, 3), dtype=np.float32)

        # normailize image by (X-X_Mean)/X_Std
        r_mean = np.mean(mouth_image_array[0:frame_n_temp, :, :, 0])
        g_mean = np.mean(mouth_image_array[0:frame_n_temp, :, :, 1])
        b_mean = np.mean(mouth_image_array[0:frame_n_temp, :, :, 2])

        r_std = np.std(mouth_image_array[0:frame_n_temp, :, :, 0])
        g_std = np.std(mouth_image_array[0:frame_n_temp, :, :, 1])
        b_std = np.std(mouth_image_array[0:frame_n_temp, :, :, 2])

        mouth_image_array[0:frame_n_temp, :, :, 0] = (mouth_image_array[0:frame_n_temp, :, :, 0] - r_mean) / r_std
        mouth_image_array[0:frame_n_temp, :, :, 1] = (mouth_image_array[0:frame_n_temp, :, :, 1] - g_mean) / g_std
        mouth_image_array[0:frame_n_temp, :, :, 2] = (mouth_image_array[0:frame_n_temp, :, :, 2] - b_mean) / b_std

        X[0, 0:frame_n_temp] = mouth_image_array[0:frame_n_temp]

        y = gestureIDs.index(int(gestureId))  # start from 0 in net class

        y_predict_probabilities = pre_model.predict(X, batch_size=None)[0]
        y_predict = y_predict_probabilities.argmax()  # start from 0 in net class

        y_predict_gestureId = gestureIDs[y_predict]

        if y_predict == y:  # id from 0 to n-1 in gestureIDs
            right_instances += 1
        total_instances += 1

        if total_instances % 100 == 0:
            print(total_instances)

    accuracy = float(right_instances) / total_instances
    print("accuracy: {}".format(accuracy))

    with open("../resource/progress/" + "group_" + str(group) + "_" + outman + "_" + str(progress) + '.txt', 'a') as fout:
        fout.write(str(accuracy) + "\n")


