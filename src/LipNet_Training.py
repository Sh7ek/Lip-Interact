from LipNet import LipNet
from LipNet_Generator import DataGenerator
from datetime import datetime
from keras.callbacks import CSVLogger, Callback
from keras import backend as K
import os

class LearningRateChanger(Callback):

    def __init__(self, patience=3, sum_patience=6, reduce_rate=0.1, verbose=1):
        super(Callback, self).__init__()
        self.patience = patience
        self.verbose = verbose
        self.sum_patience = sum_patience
        self.reduce_rate = reduce_rate
        self.best_val_score = -1.
        self.best_score = -1.
        self.wait = 0
        self.state = 0

    def on_epoch_end(self, epoch, logs=None):
        current_val_score = logs.get('val_acc')
        current_score = logs.get('acc')
        if self.state == 0 and current_val_score > 0.95 and current_score > 0.95:
            new_learning_rate = max(float(K.get_value(self.model.optimizer.lr)) * self.reduce_rate, 0.00001)
            K.set_value(self.model.optimizer.lr, new_learning_rate)
            if self.verbose > 0:
                print('\nEpoch {}: reduce learning rate to {}.'.format(epoch, new_learning_rate))
            self.state = 1
        elif self.state == 1 and current_val_score > 0.97 and current_score > 0.97:
            new_learning_rate = max(float(K.get_value(self.model.optimizer.lr)) * self.reduce_rate, 0.00001)
            K.set_value(self.model.optimizer.lr, new_learning_rate)
            if self.verbose > 0:
                print('\nEpoch {}: reduce learning rate to {}.'.format(epoch, new_learning_rate))
            self.state = 2
        elif self.state == 2 and current_val_score > 0.98:
            if self.verbose > 0:
                print('\nEpoch {}: early stopping'.format(epoch))
                self.model.stop_training = True

        # if current_score > 0.92 and current_val_score > 0.92:
        #     if current_val_score > self.best_val_score:
        #         self.best_val_score = current_val_score
        #         self.wait = 0
        #     else:
        #         self.wait += 1
        #         if self.wait >= self.sum_patience:
        #             if self.verbose > 0:
        #                 print('Epoch {}: early stopping'.format(epoch))
        #             self.model.stop_training = True
        #         elif self.wait >= self.patience:
        #             new_learning_rate = max(float(K.get_value(self.model.optimizer.lr)) * self.reduce_rate, 0.00001)
        #             K.set_value(self.model.optimizer.lr, new_learning_rate)
        #             if self.verbose > 0:
        #                 print('Epoch {}: reduce learning rate to {}.'.format(epoch, new_learning_rate))


if __name__ == '__main__':

    outman = 'wxy'
    group = 1  # 1: open apps  2: preferences  3: wechat  4: edit text  5: notification

    gestureIDs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    if group == 1:
        gestureIDs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    elif group == 2:
        gestureIDs = [11, 12, 13, 14, 15, 16, 17, 18, 42, 43]
    elif group == 3:
        gestureIDs = [19, 20, 21, 22, 23, 24, 25, 26]
    elif group == 4:
        gestureIDs = [27, 28, 29, 30, 31, 32, 33, 34, 35]
    elif group == 5:
        gestureIDs = [36, 37, 38, 39, 40, 41]

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
    epochs = 80
    class_n = len(gestureIDs)

    training_generator = DataGenerator(class_n=class_n, frames_n=70, img_h=80, img_w=100, img_c=3,
                                       batch_size=batch_size, shuffle=True, gestureIDs=gestureIDs).generate(list_training_IDs)
    validation_generator = DataGenerator(class_n=class_n, frames_n=70, img_h=80, img_w=100, img_c=3,
                                         batch_size=batch_size, shuffle=True, gestureIDs=gestureIDs).generate(list_validation_IDs)

    # create model
    lipnet = LipNet(frames_n=70, img_h=80, img_w=100, img_c=3, output_n=class_n)
    lipnet.summary()
    lipnet.compile()

    outputFolder = '../resource/new_model/group_' + str(group) + '/'
    if not os.path.exists(outputFolder):
        os.mkdir(outputFolder)

    now = datetime.now().strftime("%Y-%m-%d-%H")
    csv_logger = CSVLogger(outputFolder + outman + '_out_' + now + '_log.csv', separator=',', append=False)

    lr_changer = LearningRateChanger(patience=3, sum_patience=8, reduce_rate=0.3, verbose=1)

    lipnet.model.fit_generator(generator=training_generator,
                               steps_per_epoch=len(list_training_IDs)/batch_size,
                               epochs=epochs,
                               verbose=1,
                               callbacks=[csv_logger, lr_changer],
                               validation_data=validation_generator,
                               validation_steps=len(list_validation_IDs)/batch_size)

    lipnet.model.save(outputFolder + outman + '_out_' + now + '_model.h5')


