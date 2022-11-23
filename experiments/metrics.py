import tensorflow as tf
from tensorflow.keras import callbacks, optimizers, losses, layers
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
import pandas as pd
import numpy as np
from scipy.linalg import sqrtm
from scipy.stats import entropy
import argparse
import os


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
    return fid


class MetricsCallbacks(callbacks.Callback):
    def __init__(self, val_data, opt, params, train=False):
        super().__init__()
        self.validation_data = val_data
        self.opt = opt
        self.params_ = params
        self.train=train
        self.inception_model = self.build_inception()

    def on_train_begin(self, logs=None):
        self.IS = []
        self.FID = []

    def on_train_end(self, logs=None):
        df = pd.DataFrame(np.array([self.IS, self.FID]).T, columns=['is', 'fid'])
        df.to_csv(f'{self.opt.output_dir}/{self.opt.model}/{self.params_}_score.csv')

    def on_epoch_end(self, epoch, logs=None):
        all_preds = []
        Eb = []
        Eab = []
        for xa, xb in self.validation_data:
            # translation
            if self.opt.model =='InfoMatch':
                xab_wrapped, grids = self.model.CP(xa)
                xab, _ = self.model.R(xab_wrapped)

            elif self.opt.model == 'CycleGAN':
                xab = self.model.Gb(xa)

            elif self.opt.model == 'UNIT':
                ha, _ = self.model.Ga.encode(xa)
                xab = self.model.Gb.decode(ha)

            elif self.opt.model == 'UGATIT':
                xab, _ = self.model.Gb(xa)

            else:
                xab = self.model.G(xa)

            # preprocess
            xb = self.preprocess(xb)
            xab = self.preprocess(xab)

            # get embeddings
            _, eb = self.inception_model(xb)
            pab, eab = self.inception_model(xab)
            Eb.append(eb)
            Eab.append(eab)
            all_preds.append(pab)

        # Inception Score
        IS = []
        all_preds = tf.concat(all_preds, axis=0)
        py = tf.math.reduce_sum(all_preds, axis=0)
        for j in range(all_preds.shape[0]):
            pyx = all_preds[j, :]
            IS.append(entropy(pyx, py))
        IS = tf.exp(tf.reduce_mean(IS))
        # FID Score
        Eb = tf.concat(Eb, axis=0)
        Eab = tf.concat(Eab, axis=0)
        FID = calculate_fid(Eb.numpy(), Eab.numpy())

        # write history
        self.IS.append(IS)
        self.FID.append(FID)

        # monitor
        print(f'--fid: {FID} --is: {IS}')

    def preprocess(self, x):
        x = x * 127.5 + 127.5
        x = tf.image.resize(x, (299, 299))
        x = preprocess_input(x)
        return x

    def build_inception(self):
        inception_model = InceptionV3(include_top=True,
                                      weights="imagenet",
                                      pooling='avg')
        inception_model.trainable = False

        feature = inception_model.layers[-2].output
        prob =inception_model.layers[-1].output
        outputs = [prob, feature]
        return tf.keras.Model(inputs=inception_model.input, outputs=outputs)
