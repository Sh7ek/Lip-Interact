from LipNet import LipNet
from LipNet_Generator import DataGenerator
from datetime import datetime
from keras.callbacks import CSVLogger

if __name__ == '__main__':

    outman = 'ztx'
    # create batch generator
    training_filename = "../resource/training_list_" + outman + "_out.txt"
    with open(training_filename) as f:
        list_training_IDs = f.readlines()
        list_training_IDs = [x.strip() for x in list_training_IDs]

    validation_filename = "../resource/testing_list_" + outman + "_out.txt"
    with open(validation_filename) as f:
        list_validation_IDs = f.readlines()
        list_validation_IDs = [x.strip() for x in list_validation_IDs]

    batch_size = 32
    epochs = 60
    training_generator = DataGenerator(class_n=11, frames_n=70, img_h=80, img_w=100, img_c=3, batch_size=batch_size, shuffle=True).generate(list_training_IDs)
    validation_generator = DataGenerator(class_n=11, frames_n=70, img_h=80, img_w=100, img_c=3, batch_size=batch_size, shuffle=True).generate(list_validation_IDs)

    # create model
    lipnet = LipNet(frames_n=70, img_h=80, img_w=100, img_c=3, output_n=11)
    lipnet.summary()
    lipnet.compile()

    csv_logger = CSVLogger("../resource/csv_log_" + outman + "_out.csv", separator=',', append=False)
    lipnet.model.fit_generator(generator=training_generator,
                               steps_per_epoch=len(list_training_IDs)/batch_size,
                               epochs=epochs,
                               verbose=1,
                               callbacks=[csv_logger],
                               validation_data=validation_generator,
                               validation_steps=len(list_validation_IDs)/batch_size)

    now = datetime.now().strftime("%Y-%m-%d")
    lipnet.model.save('../resource/' + now + '_model_' + outman + '_out.h5')


