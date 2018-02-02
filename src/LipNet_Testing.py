from keras.models import load_model
import pickle
import numpy as np
import datetime

group = 9  # 1: open apps  2: preferences  3: wechat  4: edit text  5: notification
model = load_model('../resource/new_model/group_' + str(group) + '/hpk_out_2018-01-26-12_model.h5')
validation_filename = '../resource/new_training_testing_list/group_' + str(group) + '/testing_list_hpk_out.txt'

gestures = ['', 'wei xin', 'liu lan qi', 'xiang ji', 'zhi fu bao', 'yin yue', 'tao bao', 'you xiang', 'wei bo', 'nao zhong', 'di tu',
            'jie ping', 'wifi', 'jing yin', 'shou dian tong', 'tong zhi lan', 'zui jin ying yong', 'lan ya', 'suo ping',
            'peng you quan', 'sou suo', 'tian jia', 'fa zhuang tai', 'sao ma', 'dian zan', 'geng huan tou xiang', 'er wei ma',
            'che xiao', 'chong zuo', 'xiang zuo', 'xiang you', 'fu zhi', 'jian qie', 'zhan tie', 'jia cu', 'gao liang',
            'shan chu', 'cha kan', 'gua duan', 'jie ting', 'shi', 'fou', 'fan hui', 'zhuo mian']

gestureIDs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
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
    gestureIDs = [11, 12, 13, 14, 15, 16, 17, 42, 43, 36, 37, 38, 39, 40, 41]


with open(validation_filename) as f:
    list_validation_IDs = f.readlines()
    list_validation_IDs = [x.strip() for x in list_validation_IDs]

total_seconds = 0.0
total_instances = 0
right_instances = 0
# start = datetime.datetime.now()
for ID in list_validation_IDs:
    subject, gestureId, filename = ID.split("-")
    file_path = "../resource/data_new_augmented_array/" + subject + "/" + str(gestureId) + "/" + filename
    image_pkl_file = open(file_path, 'rb')
    mouth_image_array = pickle.load(image_pkl_file).astype(dtype=np.float32)  # frame_n, img_h, img_w, img_c
    image_pkl_file.close()

    start = datetime.datetime.now()

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

    y_predict_probabilities = model.predict(X, batch_size=None)[0]
    y_predict = y_predict_probabilities.argmax()  # start from 0 in net class

    y_predict_gestureId = gestureIDs[y_predict]  # start from 1

    if y_predict == y:  # id from 0 to n-1 in gestureIDs
        right_instances += 1
    else:
        print("{}  {} is recognized as {}".format(ID, gestures[int(gestureId)], gestures[y_predict_gestureId]))
        print(y_predict_probabilities)
    total_instances += 1

    if total_instances % 100 == 0:
        print(total_instances)

    end = datetime.datetime.now()
    total_seconds += (end - start).total_seconds()

# end = datetime.datetime.now()
# total_seconds = (end - start).total_seconds()

print("time: {}   instances: {}   speed: {} s".format(total_seconds, len(list_validation_IDs), total_seconds/len(list_validation_IDs)))

accuracy = float(right_instances)/total_instances
print("accuracy: {}".format(accuracy))
