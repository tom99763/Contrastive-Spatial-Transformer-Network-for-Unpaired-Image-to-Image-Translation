import sys

sys.path.append('./models')
from modules import *
from losses import *
from discriminators import Discriminator
import tensorflow as tf
from tensorflow.keras import layers


class Generator(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self.act = config['act']
        self.use_bias = config['use_bias']
        self.norm = config['norm']
        self.num_downsampls = config['num_downsamples']
        self.num_resblocks = config['num_resblocks']
        dim = config['base']

        self.blocks = tf.keras.Sequential([
            layers.Input([None, None, 3]),
            Padding2D(3, pad_type='reflect'),
            ConvBlock(dim, 7, padding='valid', use_bias=self.use_bias, norm_layer=self.norm, activation=self.act),
        ])

        for _ in range(self.num_downsampls):
            dim = dim * 2
            self.blocks.add(ConvBlock(dim, 3, strides=2, padding='same', use_bias=self.use_bias, norm_layer=self.norm,
                                      activation=self.act))

        for _ in range(self.num_resblocks):
            self.blocks.add(ResBlock(dim, 3, self.use_bias, self.norm))

        for _ in range(self.num_downsampls):
            dim = dim / 2
            self.blocks.add(ConvTransposeBlock(dim, 3, strides=2, padding='same',
                                               use_bias=self.use_bias, norm_layer=self.norm, activation=self.act))
        self.blocks.add(Padding2D(3, pad_type='reflect'))
        self.blocks.add(ConvBlock(3, 7, padding='valid', activation='tanh'))

    def call(self, x):
        return self.blocks(x)


class CycleGAN(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self.Ga = Generator(config)
        self.Gb = Generator(config)
        self.Da = Discriminator(config)
        self.Db = Discriminator(config)
        self.config = config

    def compile(self,
                Ga_optimizer,
                Gb_optimizer,
                Da_optimizer,
                Db_optimizer
                ):
        super().compile()
        self.Ga_optimizer = Ga_optimizer
        self.Gb_optimizer = Gb_optimizer
        self.Da_optimizer = Da_optimizer
        self.Db_optimizer = Db_optimizer


    @tf.function
    def train_step(self, inputs):
        xa, xb = inputs

        with tf.GradientTape(persistent=True) as tape:
            xba = self.Ga(xb)
            xab = self.Gb(xa)

            # cyclic
            xaba = self.Ga(xab)
            xbab = self.Gb(xba)

            # discrimination
            critic_real_a = self.Da(xa)
            critic_fake_a = self.Da(xba)
            critic_real_b = self.Db(xb)
            critic_fake_b = self.Da(xab)

            ###compute loss
            da_loss, ga_loss = gan_loss(critic_real_a, critic_fake_a , self.config['gan_mode'])
            db_loss, gb_loss = gan_loss(critic_real_b, critic_fake_b, self.config['gan_mode'])

            l_cycle = l1_loss(xa, xaba) + l1_loss(xb, xbab)

            l_ga = ga_loss + 10 * l_cycle
            l_gb = gb_loss + 10 * l_cycle
            l_da = da_loss
            l_db = db_loss

        Gagrads = tape.gradient(l_ga, self.Ga.trainable_weights)
        Gbgrads = tape.gradient(l_gb, self.Gb.trainable_weights)
        Dagrads = tape.gradient(l_da, self.Da.trainable_weights)
        Dbgrads = tape.gradient(l_db, self.Db.trainable_weights)

        self.Ga_optimizer.apply_gradients(zip(Gagrads, self.Ga.trainable_weights))
        self.Gb_optimizer.apply_gradients(zip(Gbgrads, self.Gb.trainable_weights))
        self.Da_optimizer.apply_gradients(zip(Dagrads, self.Da.trainable_weights))
        self.Db_optimizer.apply_gradients(zip(Dbgrads, self.Db.trainable_weights))
        return {'l_cycle': l_cycle, 'g_loss': l_ga + l_gb, 'd_loss': l_da + l_db}

    @tf.function
    def test_step(self, inputs):
        xa, xb = inputs
        xba = self.Ga(xb)
        xab = self.Gb(xa)
        # cyclic
        xaba = self.Ga(xab)
        xbab = self.Gb(xba)
        l_cycle = l1_loss(xa, xaba) + l1_loss(xb, xbab)
        return {'l_cycle': l_cycle}
