import pickle
import os
import numpy as np

outputFolder = "../resource/data_new/ztx/43/"
image_pkl_file_path = outputFolder + "frame_1_image.pkl"

if os.path.isfile(image_pkl_file_path):
    image_pkl_file = open(image_pkl_file_path, 'rb')
    mouth_image_list = pickle.load(image_pkl_file)
    valid_len = min(len(mouth_image_list), 70)
    mouth_image_list = mouth_image_list[0:valid_len]
    image_pkl_file.close()

    mouth_image_array = np.asarray(mouth_image_list, dtype=np.float32)
    print(mouth_image_array.shape)

    r_mean = np.mean(mouth_image_array[:, :, :, 0])
    g_mean = np.mean(mouth_image_array[:, :, :, 1])
    b_mean = np.mean(mouth_image_array[:, :, :, 2])

    r_std = np.std(mouth_image_array[:, :, :, 0])
    g_std = np.std(mouth_image_array[:, :, :, 1])
    b_std = np.std(mouth_image_array[:, :, :, 2])

    print('{} {} {}'.format(r_mean, g_mean, b_mean))
    print('{} {} {}'.format(r_std, g_std, b_std))

    print(mouth_image_array[10][10][60][2])
    print((mouth_image_array[10][10][60][2]-b_mean)/b_std)
    mouth_image_array[:, :, :, 0] = (mouth_image_array[:, :, :, 0] - r_mean) / r_std
    mouth_image_array[:, :, :, 1] = (mouth_image_array[:, :, :, 1] - g_mean) / g_std
    mouth_image_array[:, :, :, 2] = (mouth_image_array[:, :, :, 2] - b_mean) / b_std

    print(mouth_image_array[10][10][60][2])


