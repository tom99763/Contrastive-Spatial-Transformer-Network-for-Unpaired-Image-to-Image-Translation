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

        if not train:
            name1 = opt.source_dir.split('/')[-1]
            name2 = opt.target_dir.split('/')[-1]
            self.inception_model.load_weights(
                tf.train.latest_checkpoint(
                    f"{opt.ckpt_dir}/inception/{name1}2{name2}"))\
                .expect_partial()

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
        inception_model = InceptionV3(include_top=False,
                                      weights="imagenet",
                                      pooling='avg')
        inception_model.trainable = False

        feature = inception_model.layers[-1].output
        prob = layers.Dense(2)(feature)
        prob = tf.nn.softmax(prob, axis=-1)

        outputs = [prob, feature]

        if self.train:
            return tf.keras.Model(inputs=inception_model.input, outputs=[prob])
        else:
            return tf.keras.Model(inputs=inception_model.input, outputs=outputs)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', type=str, default='../../datasets/afhq/train/dog')
    parser.add_argument('--target_dir', type=str, default='../../datasets/afhq/train/cat')
    parser.add_argument('--ckpt_dir', type=str, default='../checkpoints')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--image_size', type=int, default=299)
    parser.add_argument('--val_size', type=int, default=0.1)
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    opt, _ = parser.parse_known_args()
    return opt

def get_image(opt, pth, label, channels = 3):
    image = tf.image.decode_jpeg(tf.io.read_file(pth), channels=channels)
    image = tf.cast(tf.image.resize(image, (opt.image_size, opt.image_size)), 'float32')
    image = preprocess_input(image)
    return image, label


def train_inception():
    opt = parse_opt()
    source_list = list(map(lambda x: f'{opt.source_dir}/{x}', os.listdir(opt.source_dir)))
    target_list = list(map(lambda x: f'{opt.target_dir}/{x}', os.listdir(opt.target_dir)))
    length = min(len(source_list), len(target_list))
    source_list = source_list[:length]
    target_list = target_list[:length]
    path_list = source_list + target_list
    labels = [0] * len(source_list) + [1] * len(target_list)

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    ds = tf.data.Dataset.from_tensor_slices((path_list, labels)).\
        map(lambda pth, label: get_image(opt, pth, label)).batch(opt.batch_size).\
        shuffle(256).prefetch(AUTOTUNE)

    metrics_callback = MetricsCallbacks('none', opt, 'none', train=True)
    model = metrics_callback.inception_model
    model.compile(optimizer = optimizers.Adam(learning_rate = opt.lr),
                  loss = losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics = 'acc',
                  )

    name1 = opt.source_dir.split('/')[-1]
    name2 = opt.target_dir.split('/')[-1]
    save_dir = f"{opt.ckpt_dir}/inception/{name1}2{name2}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    checkpoint_callback = callbacks.ModelCheckpoint(
        filepath=f"{save_dir}/{name1}2{name2}", save_weights_only=True)
    model.fit(ds, callbacks = [checkpoint_callback], epochs=opt.num_epochs)

if __name__ == '__main__':
    train_inception()








