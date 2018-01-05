from threading import Thread
from time import sleep
import numpy as np
from keras.models import load_model
import tensorflow as tf


class RecognizeThread:
    def __init__(self):
        self.newArrival = False
        self.stopped = False
        self.list_mouth_frame = []
        self.X = np.zeros(shape=(1, 70, 80, 100, 3), dtype=np.float32)
        self.recognizeDone = False
        self.y_predict = 0
        self.model = load_model('../resource/model/2017-12-01_model.h5')
        self.model._make_predict_function()
        self.graph = tf.get_default_graph()
        self.model_loaded = True
        print("------ LOAD MODEL COMPLETE ------")

    def start(self):
        Thread(target=self.worker, args=()).start()
        return self

    def worker(self):
        while not self.stopped:
            if self.newArrival:
                frame_n_temp = min(len(self.list_mouth_frame), 70)
                self.X = np.zeros(shape=(1, 70, 80, 100, 3), dtype=np.float32)
                self.X[0, 0:frame_n_temp] = np.asarray(self.list_mouth_frame, dtype=np.float32)[0:frame_n_temp]

                with self.graph.as_default():
                    y_predict_probabilities = self.model.predict(self.X, batch_size=None)[0]
                self.y_predict = y_predict_probabilities.argmax() + 1

                # print("Mouth Frame Valid Length: {}".format(len(self.list_mouth_frame)))
                # print("Recognition Result: {}".format(self.y_predict))
                self.recognizeDone = True
                self.newArrival = False

            sleep(0.005)

    def stop(self):
        self.stopped = True

    def setData(self, list_user_mouth_frame):
        self.list_mouth_frame = list_user_mouth_frame[:]
        self.newArrival = True

    def reset_recognize_state(self):
        self.recognizeDone = False

    def recognize_complete(self):
        return self.recognizeDone
