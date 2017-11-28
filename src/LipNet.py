from keras import backend as K
from keras.layers import Input
from keras.layers.convolutional import ZeroPadding3D, Conv3D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation, SpatialDropout3D, Flatten, Dense
from keras.layers.pooling import MaxPooling3D
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.recurrent import GRU
from keras.models import Model
from keras.optimizers import Adam

class LipNet(object):
    def __init__(self, img_c=3, img_w=100, img_h=80, frames_n=70, output_n=10):
        self.img_c = img_c
        self.img_w = img_w
        self.img_h = img_h
        self.frames_n = frames_n
        self.output_n = output_n
        self.build()

    def build(self):
        if K.image_data_format() == 'channels_first':
            input_shape = (self.img_c, self.frames_n, self.img_h, self.img_w)
        else:
            input_shape = (self.frames_n, self.img_h, self.img_w, self.img_c)

        self.input_data = Input(name='input', shape=input_shape, dtype='float32')

        # 70 x 100 x 80 x 3
        self.zero1 = ZeroPadding3D(padding=(1, 2, 2), name='zero1')(self.input_data)
        self.conv1 = Conv3D(32, (3, 5, 5), strides=(1, 2, 2), kernel_initializer='he_normal', name='conv_1')(self.zero1)
        # 70 x 50 x 40 x 32
        self.batch1 = BatchNormalization(name='batch1')(self.conv1)
        self.actv1 = Activation('relu', name='actv1')(self.batch1)
        self.maxp1 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max1')(self.actv1)
        # 70 x 25 x 20 x 32
        self.drop1 = SpatialDropout3D(0.5)(self.maxp1)

        self.zero2 = ZeroPadding3D(padding=(1, 2, 2), name='zero2')(self.drop1)
        self.conv2 = Conv3D(64, (3, 5, 5), strides=(1, 1, 1), kernel_initializer='he_normal', name='conv_2')(self.zero2)
        # 70 x 25 x 20 x 64
        self.batch2 = BatchNormalization(name='batch2')(self.conv2)
        self.actv2 = Activation('relu', name='actv2')(self.batch2)
        self.maxp2 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max2')(self.actv2)
        # 70 x 12 x 10 x 64
        self.drop2 = SpatialDropout3D(0.5)(self.maxp2)

        self.zero3 = ZeroPadding3D(padding=(1, 1, 1), name='zero3')(self.drop2)
        self.conv3 = Conv3D(96, (3, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', name='conv3')(self.zero3)
        # 70 x 12 x 10 x 96
        self.batch3 = BatchNormalization(name='batch3')(self.conv3)
        self.actv3 = Activation('relu', name='actv3')(self.batch3)
        self.maxp3 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max3')(self.actv3)
        # 70 x 6 x 5 x 96
        self.drop3 = SpatialDropout3D(0.5)(self.maxp3)

        self.resh1 = TimeDistributed(Flatten())(self.drop3)
        # 70 x (6 x 5 x 96) = 70 x 2880

        # self.gru_1 = Bidirectional(GRU(256, return_sequences=True, kernel_initializer='Orthogonal', name='gru1'), merge_mode='concat')(self.resh1)
        # self.gru_2 = Bidirectional(GRU(256, kernel_initializer='Orthogonal', name='gru2'), merge_mode='concat')(self.gru_1)
        self.gru_1 = Bidirectional(GRU(256, kernel_initializer='Orthogonal', name='gru1'), merge_mode='concat')(self.resh1)

        # 70 x (256 x 2) = 70 x 512

        # transforms RNN output to classification:
        self.prediction = Dense(units=self.output_n, activation='softmax', kernel_initializer='he_normal', name='predict')(self.gru_1)
        # 70 x (10)

        self.model = Model(inputs=self.input_data, outputs=self.prediction)

    def summary(self):
        Model(inputs=self.input_data, outputs=self.prediction).summary()

    def compile(self):
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        self.model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])














