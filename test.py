from NormalizingAutoencoder import NormalizingAutoencoder

if __name__ == "__main__":
    test_net = NormalizingAutoencoder(1, 16, "auto_encoder2-200.hdf5")

    en_out = test_net.model_encoder.predict([5000000])
    print(en_out)
    de_out = test_net.model_decoder.predict(en_out)
    print(de_out)
