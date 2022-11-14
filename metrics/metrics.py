import tensorflow as tf
from tensorflow.keras import callbacks
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
import pandas as pd
import sys
import numpy as np
from scipy.linalg import sqrtm

def calculate_fid(Eb, Eab):
    # calculate mean and covariance statistics
    mu1, sigma1 = Eb.mean(axis=0), np.cov(Eb, rowvar=False)
    mu2, sigma2 = Eab.mean(axis=0), np.cov(Eab, rowvar=False)

    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)

    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))

    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return 

class MetricsCallbacks(callbacks.Callback):
    def __init__(self, val_data, opt, params):
        super().__init__()
        self.validation_data = val_data
        self.opt=opt
        self.params_ = params
        self.inception_model = InceptionV3(include_top=False,
                                           weights="imagenet",
                                           pooling='avg')
        self.inception_model.trainable = False

    def on_train_begin(self, logs=None):
        self.FID = []

    def on_train_end(self, logs=None):
        df = pd.DataFrame(self.FID, columns = ['fid'])
        df.to_csv(f'{self.opt.output_dir}/{self.opt.model}/{self.params_}_score.csv')

    def on_epoch_end(self, epoch, logs=None):
        Eb = []
        Eab = []
        for xa, xb in self.validation_data:
            #translation
            xab = self.model.G(xa)

            #preprocess
            xb = self.preprocess(xb)
            xab = self.preprocess(xab)

            #get embeddings
            eb = self.inception_model(xb)
            eab = self.inception_model(xab)
            Eb.append(eb)
            Eab.append(eab)
        Eb = tf.concat(Eb, axis=0)
        Eab = tf.concat(Eab, axis=0)
        FID = calculate_fid(Eb.numpy(), Eab.numpy())
        self.FID.append(FID)
        print(f'epoch: {epoch} -- fid: {FID}')

    def preprocess(self, x):
        x = x * 127.5 + 127.5
        x = tf.image.resize(x, (299, 299))
        x = preprocess_input(x)
        return x
