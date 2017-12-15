from threading import Thread
from time import sleep
import pickle
import os

class Write2File:
    def __init__(self, subject, gestureID):
        self.newArrival = False
        self.stopped = False
        self.mouthFrameList = []
        self.writeDone = False
        self.outputFileIndex = 0
        self.outputFolder = "../resource/data_new/" + subject + "/" + str(gestureID) + "/"
        if not os.path.exists(self.outputFolder):
            os.mkdir(self.outputFolder)
        print("Write To File Init ...")

    def start(self):
        Thread(target=self.worker, args=()).start()
        return self

    def worker(self):
        while not self.stopped:
            if self.newArrival:
                # write mouthFrameList to file
                outputFileName = self.outputFolder + "frame_" + str(self.outputFileIndex) + "_image.pkl"
                output = open(outputFileName, 'wb')
                pickle.dump(self.mouthFrameList, output)
                output.close()

                # print("write to file done {}".format(len(self.mouthFrameList)))
                self.writeDone = True
                self.newArrival = False
            sleep(0.005)

    def stop(self):
        self.stopped = True

    def setData(self, userMouthFrameList, count):
        self.mouthFrameList = userMouthFrameList[:]
        self.outputFileIndex = count
        self.newArrival = True

    def setNotDone(self):
        self.writeDone = False

    def getDone(self):
        return self.writeDone
