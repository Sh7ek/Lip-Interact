import numpy as np
import pickle
from keras.preprocessing import sequence
# import cv2


class DataGenerator(object):
    # If the shuffle option is True, the generator shuffles the order of exploration of the samples before each new epoch.
    def __init__(self, class_n=10, frames_n=70, img_h=80, img_w=100, img_c=3, batch_size=32, shuffle=True, gestureIDs=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]):
        self.class_n = class_n
        self.frames_n = frames_n
        self.img_h = img_h
        self.img_w = img_w
        self.img_c = img_c
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.gestureIDs = gestureIDs

    # generate the next batch of data
    def generate(self, list_IDs):
        """Generates batches of samples"""
        while 1:
            # Generate order of exploration of dataset
            indexes = self.__get_index_order(list_IDs)

            # Generate batches
            imax = int(len(indexes) / self.batch_size)
            for i in range(imax):
                list_IDs_i = [list_IDs[k] for k in indexes[i * self.batch_size : (i+1)*self.batch_size]]
                # Generate the data
                X, y = self.__data_generation(list_IDs_i)
                yield X, y

        # indexes = self.__get_index_order(list_IDs)
        # i = 1
        # list_IDs_i = [list_IDs[k] for k in indexes[i * self.batch_size: (i + 1) * self.batch_size]]
        # # Generate the data
        # X, y = self.__data_generation(list_IDs_i)
        # return X, y


    def __get_index_order(self, list_IDs):
        """Generates order of exploration"""
        indexes = np.arange(len(list_IDs))
        if self.shuffle == True:
            np.random.shuffle(indexes)
        return indexes


    def __data_generation(self,list_IDs_temp):
        """Generates data of batch_size samples"""
        # X : (n_samples, frame_n, img_h, img_w, img_c)
        X = np.zeros((self.batch_size, self.frames_n, self.img_h, self.img_w, self.img_c), dtype=np.float32)
        y = np.zeros((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # ID, aka file name format : "sk_flip-7-frame_11_image.pkl"
            subject, gestureId, filename = ID.split("-")

            file_path = "../resource/data_new_augmented_array/" + subject + "/" + str(gestureId) + "/" + filename
            image_pkl_file = open(file_path, 'rb')
            mouth_image_array = pickle.load(image_pkl_file).astype(dtype=np.float32)  # frame_n, img_h, img_w, img_c
            image_pkl_file.close()

            frame_n_temp = min(mouth_image_array.shape[0], self.frames_n)

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

            y[i] = self.gestureIDs.index(int(gestureId))   # start from 0

        return X, self.sparsify(y)


    def sparsify(self, y):
        """Returns labels in binary NumPy array"""
        # note that this sparsify function is adapted for labels starting at 0. If your labels start at 1, simply change the expression y[i] == j to y[i] == j+1 in the piece of code above.
        return np.array([[1 if y[i] == j else 0 for j in range(self.class_n)] for i in range(y.shape[0])])


# if __name__ == "__main__":
#     generator = DataGenerator(class_n=11, frames_n=70, img_h=80, img_w=100, img_c=3, batch_size=32, shuffle=True)
#     training_filename = "../resource/training_list.txt"
#     with open(training_filename) as f:
#         list_IDs = f.readlines()
#     list_IDs = [x.strip() for x in list_IDs]
#     print("{} {}".format(len(list_IDs), list_IDs[0]))
#     X, y = generator.generate(list_IDs)
#
#     print(y)
#     print(X.shape)
#
#     cv2.namedWindow("frame")
#     cv2.moveWindow("frame", 1000, 400)
#     for k in range(X.shape[0]):
#         print(k)
#         sample = X[k].astype(np.uint8)  # (70, 80, 100, 3)  float32
#         for i in range(0, sample.shape[0]):
#             image = sample[i]
#             cv2.imshow('frame', image)
#             key = cv2.waitKey(40) & 0xFF
#
#         key = cv2.waitKey(0) & 0xFF
#         if key == ord('q'):
#             break
#
#     cv2.destroyAllWindows()



