import sys
sys.path.append('./models')
from modules import *
from losses import *
import tensorflow as tf
from tensorflow.keras import layers
from discriminators import *


class VectorQuantizer(layers.Layer):
    def __init__(self, config):
        super().__init__()
        self.n_embs = config['n_embs']
        self.beta = config['beta']

    def build(self, shape):
        emb_dim = shape[-1]
        init = tf.random_uniform_initializer()
        self.embs = tf.Variable(
            initial_value=init(shape=[self.n_embs, emb_dim], dtype='float32'),
            trainable=True,
            name='embeddings'
        )

    def call(self, ze):
        b, h, w, c = ze.shape
        z = tf.reshape(ze, [-1, c])
        idx = self.get_code_idx(z)
        zq = tf.gather(self.embs, idx, axis=0)
        zq = tf.reshape(zq, [b, h, w, c])

        # loss
        commitment_loss = self.beta * tf.reduce_mean((tf.stop_gradient(zq) - ze) ** 2)
        coodbook_loss = tf.reduce_mean((zq - tf.stop_gradient(ze)) ** 2)
        self.add_loss(commitment_loss + coodbook_loss)

        # straight-through estimator to update the encoder
        zq = ze + tf.stop_gradient(zq - ze)
        return zq

    def get_code_idx(self, z):
        # distance metric
        similarity = z @ tf.transpose(self.embs)  # [bhw,n_embs]
        distances = tf.reduce_sum(z ** 2, axis=-1, keepdims=True) \
                    + tf.expand_dims(tf.reduce_sum(self.embs ** 2, axis=-1), axis=0) \
                    - 2 * similarity
        # align codes
        idx = tf.argmin(distances, axis=-1)  # [bhw]
        return idx


class Generator(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self.act = config['act']
        self.use_bias = config['use_bias']
        self.norm = config['norm']
        self.num_downsampls = config['num_downsamples']
        self.num_resblocks = config['num_resblocks']
        dim = config['base']

        self.encoder = tf.keras.Sequential([
            layers.Input([None, None, 3]),
            Padding2D(3, pad_type='reflect'),
            ConvBlock(dim, 7, padding='valid', use_bias=self.use_bias, norm_layer=self.norm, activation=self.act),
        ])

        for _ in range(self.num_downsampls):
            dim = dim * 2
            self.encoder.add(ConvBlock(dim, 3, strides=2, padding='same', use_bias=self.use_bias, norm_layer=self.norm,
                                       activation=self.act))

        self.resblocks = tf.keras.Sequential()
        for _ in range(self.num_resblocks):
            self.resblocks.add(ResBlock(dim, 3, self.use_bias, self.norm))

        self.decoder = tf.keras.Sequential()
        for _ in range(self.num_downsampls):
            dim = dim / 2
            self.decoder.add(
                ConvTransposeBlock(dim, 3, strides=2, padding='same', use_bias=self.use_bias, norm_layer=self.norm,
                                   activation=self.act))
        self.decoder.add(Padding2D(3, pad_type='reflect'))
        self.decoder.add(ConvBlock(3, 7, padding='valid', activation='tanh'))

        self.quantizer = VectorQuantizer(config)

    def call(self, x):
        x = self.encoder(x)
        x = self.quantizer(x)
        x = self.resblocks(x)
        x = self.decoder(x)
        return x


class VQCycleGAN(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self.Ga = Generator(config)
        self.Gb = Generator(config)
        self.Da = Discriminator(config)
        self.Db = Discriminator(config)
        self.config = config

    def compile(self,
                G_optimizer,
                D_optimizer):
        super(VQCycleGAN, self).compile()
        self.G_optimizer = G_optimizer
        self.D_optimizer = D_optimizer

    @tf.function
    def train_step(self, inputs):
        xa, xb = inputs
        with tf.GradientTape(persistent=True) as tape:
            # generation
            xaa = self.Ga(xa)
            xbb = self.Gb(xb)
            xab = self.Gb(xa)
            xba = self.Ga(xb)
            xaba = self.Ga(xab)
            xbab = self.Gb(xba)

            # discrimination
            critic_a_real = self.Da(xa)
            critic_b_real = self.Db(xb)
            critic_a_fake = self.Da(xba)
            critic_b_fake = self.Db(xab)

            ###compute loss
            l_idt = l1_loss(xa, xaa) + l1_loss(xb, xbb)
            l_cyc = l1_loss(xa, xaba) + l1_loss(xb, xbab)
            l_da, l_ga = gan_loss(critic_a_real, critic_a_fake, self.config['gan_mode'])
            l_db, l_gb = gan_loss(critic_b_real, critic_b_fake, self.config['gan_mode'])
            l_vq = sum(self.Ga.losses) + sum(self.Gb.losses)

            g_loss = self.config['lambda_vq'] *  l_vq + self.config['lambda_idt'] * l_idt + self.config['lambda_cyc'] * l_cyc + l_ga + l_gb
            d_loss = l_da + l_db

        g_grads = tape.gradient(g_loss, self.Ga.trainable_weights + self.Gb.trainable_weights)
        d_grads = tape.gradient(d_loss, self.Da.trainable_weights + self.Db.trainable_weights)
        self.G_optimizer.apply_gradients(zip(g_grads, self.Ga.trainable_weights + self.Gb.trainable_weights))
        self.D_optimizer.apply_gradients(zip(d_grads, self.Da.trainable_weights + self.Db.trainable_weights))

        return {'vq': l_vq, 'identity': l_idt, 'cycle': l_cyc, 'g_loss': l_ga + l_gb, 'd_loss': l_da + l_db}

    @tf.function
    def test_step(self, inputs):
        xa, xb = inputs
        xaa = self.Ga(xa)
        xbb = self.Gb(xb)
        xab = self.Gb(xa)
        xba = self.Ga(xb)
        xaba = self.Ga(xab)
        xbab = self.Gb(xba)

        ###compute loss
        l_idt = l1_loss(xa, xaa) + l1_loss(xb, xbb)
        l_cyc = l1_loss(xa, xaba) + l1_loss(xb, xbab)
        l_vq = sum(self.Ga.losses) + sum(self.Gb.losses)
        return {'vq': l_vq, 'identity': l_idt, 'cycle': l_cyc}
