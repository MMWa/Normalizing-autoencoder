import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.layers import *
from keras_tqdm import TQDMCallback

from NormalizingAutoencoder import NormalizingAutoencoder

if __name__ == "__main__":
    test_net = NormalizingAutoencoder(1, 16, None)

    # train on random numbers-------------------
    df_train = pd.DataFrame(np.random.uniform(0, 100000, size=(8000,)))
    filepath = "auto_encoder1-{epoch:02d}.hdf5"
    checkpoint = ModelCheckpoint(filepath, verbose=0, period=100)
    test_net.model_ae.fit(df_train, df_train, nb_epoch=1500, batch_size=50000, shuffle=True, verbose=0,
                          callbacks=[TQDMCallback(), checkpoint])

    test_net = NormalizingAutoencoder(1, 16, "auto_encoder1-1500.hdf5")

    # train on random numbers-------------------
    df_train = pd.DataFrame(np.random.uniform(0, 10000000, size=(800000,)))
    filepath = "auto_encoder2-{epoch:02d}.hdf5"
    checkpoint = ModelCheckpoint(filepath, verbose=0, period=100)
    test_net.model_ae.fit(df_train, df_train, nb_epoch=200, batch_size=50000, shuffle=True, verbose=0,
                          callbacks=[TQDMCallback(), checkpoint])

    test_net = NormalizingAutoencoder(1, 16, "auto_encoder2-200.hdf5")
