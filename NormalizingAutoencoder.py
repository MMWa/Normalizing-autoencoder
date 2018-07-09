from keras.models import Model
from keras.layers import Dense, Lambda, Input, K
from keras.optimizers import TFOptimizer

import tensorflow as tf


class NormalizingAutoencoder:
    def __init__(self, data_shape, choke_shape, filename=None):
        self.data_shape = data_shape
        self.choke_shape = choke_shape

        self.__define()

        self.load_file(self.model_ae, filename)

        optimizer = TFOptimizer(tf.train.AdadeltaOptimizer())

        self.model_encoder.compile(optimizer=optimizer, loss=self.rmsle, metrics=[self.rmsle])
        self.model_decoder.compile(optimizer=optimizer, loss=self.rmsle, metrics=[self.rmsle])
        self.model_ae.compile(optimizer=optimizer, loss=self.rmsle, metrics=[self.rmsle])

    def __define(self):
        self.encoder_input, self.encoder = self.__define_encoder()
        self.decoder_input, self.decoder = self.__define_decoder()

        self.model_encoder = Model(inputs=self.encoder_input, outputs=self.encoder)
        self.model_decoder = Model(inputs=self.decoder_input, outputs=self.decoder)

        self.full_out = self.model_decoder(self.model_encoder(self.encoder_input))

        self.model_ae = Model(inputs=self.encoder_input, outputs=self.full_out)

    def __define_encoder(self):
        encoder_input = Input(shape=(self.data_shape,))
        encoder = Lambda(self.log1p)(encoder_input)
        encoder = Dense(128, activation='linear')(encoder)
        encoder = Dense(64, activation='linear')(encoder)
        encoder = Dense(32, activation='linear')(encoder)
        encoder = Dense(self.choke_shape, activation='relu')(encoder)
        return encoder_input, encoder

    def __define_decoder(self):
        decoder_input = Input(shape=(self.choke_shape,))
        decoder = Dense(self.choke_shape, activation='linear')(decoder_input)
        decoder = Dense(32, activation='linear')(decoder)
        decoder = Dense(64, activation='linear')(decoder)
        decoder = Dense(128, activation='linear')(decoder)
        decoder = Dense(self.data_shape, activation='linear')(decoder)
        decoder = Lambda(self.expm1)(decoder)
        return decoder_input, decoder

    def log1p(self, x):
        return tf.log1p(x)

    def expm1(self, x):
        return tf.expm1(x)

    def rmse(self, y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

    def rmsle(self, y_true, y_pred):
        return K.sqrt(K.mean(K.square(tf.log1p(y_true) - tf.log1p(y_pred))))

    def load_file(self, model, filename):
        try:
            model.load_weights(filename)
        except:
            print("error loading " + str(filename))
