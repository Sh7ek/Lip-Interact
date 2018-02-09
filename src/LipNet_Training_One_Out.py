from LipNet import LipNet
from LipNet_Generator import DataGenerator
from datetime import datetime
from keras.callbacks import CSVLogger, Callback
from keras import backend as K
from keras.optimizers import Adam
import os
import sys

class LearningRateChanger(Callback):

    def __init__(self, kind=1, verbose=1):
        super(Callback, self).__init__()
        self.verbose = verbose
        self.best_val_score = -1.
        self.kind = kind
        self.wait = 0
        self.second_start = 11
        self.second_end = 11
        self.third_start = 12
        self.patience = 10
        if self.kind == 2:
            self.second_start = 10
            self.second_end = 18
            self.third_start = 19
            self.patience = 10

    def on_epoch_end(self, epoch, logs=None):
        current_val_score = logs.get('val_acc')
        self.best_val_score = max(self.best_val_score, current_val_score)

        if self.second_start <= epoch <= self.second_end:
            new_learning_rate = 0.0003
            K.set_value(self.model.optimizer.lr, new_learning_rate)
        elif epoch >= self.third_start:
            new_learning_rate = 0.0001
            K.set_value(self.model.optimizer.lr, new_learning_rate)
            if current_val_score > self.best_val_score:
                self.best_val_score = current_val_score
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    if self.verbose > 0:
                        print('\nEpoch {}: early stopping'.format(epoch))
                    self.model.stop_training = True


if __name__ == '__main__':

    outman = 'hpk'
    group = 6  # 1: open apps  2: preferences  3: wechat  4: edit text  5: notification
    lr_type = 1

    if len(sys.argv) == 4:
        print(str(sys.argv))
        group = int(sys.argv[1])
        outman = sys.argv[2]
        lr_type = int(sys.argv[3])

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

    ttFolder = '../resource/new_training_testing_list/group_' + str(group) + '/'
    training_filename = ttFolder + 'training_list_' + outman + '_out.txt'
    with open(training_filename) as f:
        list_training_IDs = f.readlines()
        list_training_IDs = [x.strip() for x in list_training_IDs]

    validation_filename = ttFolder + 'testing_list_' + outman + '_out.txt'
    with open(validation_filename) as f:
        list_validation_IDs = f.readlines()
        list_validation_IDs = [x.strip() for x in list_validation_IDs]

    batch_size = 32
    epochs = 45
    class_n = len(gestureIDs)

    training_generator = DataGenerator(class_n=class_n, frames_n=70, img_h=80, img_w=100, img_c=3,
                                       batch_size=batch_size, shuffle=True, gestureIDs=gestureIDs).generate(list_training_IDs)
    validation_generator = DataGenerator(class_n=class_n, frames_n=70, img_h=80, img_w=100, img_c=3,
                                         batch_size=batch_size, shuffle=True, gestureIDs=gestureIDs).generate(list_validation_IDs)

    # create model
    lipnet = LipNet(frames_n=70, img_h=80, img_w=100, img_c=3, output_n=class_n)
    # lipnet.summary()

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    lipnet.model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    outputFolder = '../resource/new_model/group_' + str(group) + '/'
    if not os.path.exists(outputFolder):
        os.mkdir(outputFolder)

    now = datetime.now().strftime("%Y-%m-%d-%H")
    csv_logger = CSVLogger(outputFolder + outman + '_out_' + now + '_log.csv', separator=',', append=False)

    lr_changer = LearningRateChanger(kind=lr_type, verbose=1)

    lipnet.model.fit_generator(generator=training_generator,
                               steps_per_epoch=len(list_training_IDs)/batch_size,
                               epochs=epochs,
                               verbose=1,
                               callbacks=[csv_logger, lr_changer],
                               validation_data=validation_generator,
                               validation_steps=len(list_validation_IDs)/batch_size)

    lipnet.model.save(outputFolder + outman + '_out_' + now + '_model.h5')


