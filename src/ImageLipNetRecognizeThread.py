from threading import Thread
from time import sleep
import numpy as np
from keras.models import load_model
import tensorflow as tf


class RecognizeThread:
    group = 8
    gestureIDs_group_2 = [11, 12, 13, 14, 15, 16, 17, 42, 43]
    gestureIDs_group_6 = [11, 12, 13, 14, 15, 16, 17, 42, 43, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    gestureIDs_group_7 = [11, 12, 13, 14, 15, 16, 17, 42, 43, 19, 20, 22, 23, 24, 25, 26]
    gestureIDs_group_8 = [11, 12, 13, 14, 15, 16, 17, 42, 43, 27, 28, 29, 30, 31, 33, 34, 35]
    gestureIDs_group_9 = [11, 12, 13, 14, 15, 16, 17, 42, 43, 36, 37]
    gestureNames = ['WECHAT', 'BROWSER', 'CAMERA', 'ALIPAY', 'MUSIC', 'TAOBAO', 'MAIL', 'WEIBO', 'CLOCK', 'MAP',
                    'SCREENSHOT', 'WIFI', 'SOUND', 'FLASHLIGHT', 'NOTIFICATION', 'RECENT', 'BLUETOOTH', 'LOCK',
                    'MOMENTS', 'SEARCH', 'ADD', 'POST', 'SCANCODE', 'LIKE', 'COLLECTION', 'SHOWCODE',
                    'UNDO', 'REDO', 'LEFT', 'RIGHT', 'COPY', 'CUT', 'PASTE', 'BOLD', 'HIGHLIGHT',
                    'DELETE', 'LOOK', 'ANSWER', 'REFUSE', 'YES', 'NO',
                    'BACK', 'HOME']

    def __init__(self, socketServer):
        self.socketServer = socketServer
        self.newArrival = False
        self.stopped = False
        self.list_mouth_frame = []
        self.X = np.zeros(shape=(1, 70, 80, 100, 3), dtype=np.float32)
        self.y_predict = 0

        self.model_group2 = load_model('../resource/new_model/group_2/_out_2018-01-31-11_model.h5')
        self.model_group2._make_predict_function()
        print("------ MODEL 1 LOADED ------")
        self.model_group6 = load_model('../resource/new_model/group_6/_out_2018-01-31-13_model.h5')
        self.model_group6._make_predict_function()
        print("------ MODEL 2 LOADED ------")
        self.model_group7 = load_model('../resource/new_model/group_7/_out_2018-01-31-17_model.h5')
        self.model_group7._make_predict_function()
        print("------ MODEL 3 LOADED ------")
        self.model_group8 = load_model('../resource/new_model/group_8/_out_2018-01-31-20_model.h5')
        self.model_group8._make_predict_function()
        print("------ MODEL 4 LOADED ------")
        self.model_group9 = load_model('../resource/new_model/group_9/_out_2018-01-31-23_model.h5')
        self.model_group9._make_predict_function()
        print("------ MODEL 5 LOADED ------")
        self.graph = tf.get_default_graph()
        self.model_loaded = True
        print("------ LOAD MODEL COMPLETE ------")

    def start(self):
        Thread(target=self.worker, args=()).start()
        return self

    def worker(self):
        while not self.stopped:
            if self.newArrival:
                self.newArrival = False
                frame_n_temp = min(len(self.list_mouth_frame), 70)
                self.X = np.zeros(shape=(1, 70, 80, 100, 3), dtype=np.float32)
                if self.list_mouth_frame[0].shape[0] == 80 and self.list_mouth_frame[0].shape[1] == 100:
                    self.X[0, 0:frame_n_temp] = np.asarray(self.list_mouth_frame, dtype=np.float32)[0:frame_n_temp]

                    # normailize image by (X-X_Mean)/X_Std
                    r_mean = np.mean(self.X[0, 0:frame_n_temp, :, :, 0])
                    g_mean = np.mean(self.X[0, 0:frame_n_temp, :, :, 1])
                    b_mean = np.mean(self.X[0, 0:frame_n_temp, :, :, 2])

                    r_std = np.std(self.X[0, 0:frame_n_temp, :, :, 0])
                    g_std = np.std(self.X[0, 0:frame_n_temp, :, :, 1])
                    b_std = np.std(self.X[0, 0:frame_n_temp, :, :, 2])

                    self.X[0, 0:frame_n_temp, :, :, 0] = (self.X[0, 0:frame_n_temp, :, :, 0] - r_mean) / r_std
                    self.X[0, 0:frame_n_temp, :, :, 1] = (self.X[0, 0:frame_n_temp, :, :, 1] - g_mean) / g_std
                    self.X[0, 0:frame_n_temp, :, :, 2] = (self.X[0, 0:frame_n_temp, :, :, 2] - b_mean) / b_std

                    if RecognizeThread.group == 2:
                        with self.graph.as_default():
                            y_predict_probabilities = self.model_group2.predict(self.X, batch_size=None)[0]
                            self.y_predict = y_predict_probabilities.argmax()
                            gestureName = RecognizeThread.gestureNames[RecognizeThread.gestureIDs_group_2[self.y_predict]-1]
                            self.socketServer.SendToAllConnections(gestureName + '\n')
                            print("Recognition Result: {}".format(gestureName))
                    elif RecognizeThread.group == 6:
                        with self.graph.as_default():
                            y_predict_probabilities = self.model_group6.predict(self.X, batch_size=None)[0]
                            self.y_predict = y_predict_probabilities.argmax()
                            gestureName = RecognizeThread.gestureNames[RecognizeThread.gestureIDs_group_6[self.y_predict]-1]
                            self.socketServer.SendToAllConnections(gestureName + '\n')
                            print("Recognition Result: {}".format(gestureName))
                    elif RecognizeThread.group == 7:
                        with self.graph.as_default():
                            y_predict_probabilities = self.model_group7.predict(self.X, batch_size=None)[0]
                            self.y_predict = y_predict_probabilities.argmax()
                            gestureName = RecognizeThread.gestureNames[RecognizeThread.gestureIDs_group_7[self.y_predict]-1]
                            self.socketServer.SendToAllConnections(gestureName + '\n')
                            print("Recognition Result: {}".format(gestureName))
                    elif RecognizeThread.group == 8:
                        with self.graph.as_default():
                            y_predict_probabilities = self.model_group8.predict(self.X, batch_size=None)[0]
                            self.y_predict = y_predict_probabilities.argmax()
                            gestureName = RecognizeThread.gestureNames[RecognizeThread.gestureIDs_group_8[self.y_predict]-1]
                            self.socketServer.SendToAllConnections(gestureName + '\n')
                            print("Recognition Result: {}".format(gestureName))
                    elif RecognizeThread.group == 9:
                        with self.graph.as_default():
                            y_predict_probabilities = self.model_group9.predict(self.X, batch_size=None)[0]
                            self.y_predict = y_predict_probabilities.argmax()
                            gestureName = RecognizeThread.gestureNames[RecognizeThread.gestureIDs_group_9[self.y_predict]-1]
                            self.socketServer.SendToAllConnections(gestureName + '\n')
                            print("Recognition Result: {}".format(gestureName))
                    else:
                        print("Group Error")

            sleep(0.002)

    def stop(self):
        self.stopped = True

    def setData(self, list_user_mouth_frame):
        self.list_mouth_frame = list_user_mouth_frame[:]
        self.newArrival = True

    def reset_recognize_state(self):
        self.recognizeDone = False

    def recognize_complete(self):
        return self.recognizeDone
