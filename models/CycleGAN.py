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
      dim = dim  * 2
      self.blocks.add(ConvBlock(dim, 3, strides=2, padding='same', use_bias=self.use_bias, norm_layer=self.norm, activation=self.act))
      
    for _ in range(self.num_resblocks):
      self.blocks.add(ResBlock(dim, 3, self.use_bias, self.norm))
      
    for _ in range(self.num_downsampls):
      dim  = dim / 2
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
                G_optimizer,
                F_optimizer,
                D_optimizer):
        super().compile()
        self.G_optimizer = G_optimizer
        self.F_optimizer = F_optimizer
        self.D_optimizer = D_optimizer

    @tf.function
    def train_step(self, inputs):
        xa, xb = inputs
        
        with tf.GradientTape(persistent=True) as tape:
          xaa = self.Ga(xa)
          xba = self.Ga(xb)
          xbb = self.Gb(xb)
          xab = self.Gb(xa)
          
          #cyclic
          xaba = self.Ga(xab)
          xbab = self.Gb(xba)
          
          #discrimination
          critic_real_a = self.Da(xa)
          critic_fake_a = self.Da(xba)
          critic_real_b = self.Db(xb)
          critic_fake_b = self.Da(xab)
          
          critic_real = tf.concat([critic_real_a, critic_real_b], axis=0)
          critic_fake = tf.concat([critic_fake_a, critic_fake_b], axis=0)
          
          ###compute loss
          d_loss, g_loss = gan_loss(critic_real, critic_fake, self.config['gan_mode'])
          l_r = l1_loss(xa, xaa) + l1_loss(xb, xbb)
          l_cycle = l1_loss(xa, xaba) + l1_loss(xb, xbab)
          l_g = g_loss + l_r + l_cycle
          l_d = d_loss
          
        Ggrads = tape.gradient(l_g, self.Ga.trainable_weights + self.Gb.trainable_weights)
        Dgrads = tape.gradient(l_d, self.Da.trainable_weights + self.Db.trainable_weights)
        self.G_optimizer.apply_gradients(zip(Ggrads, self.Ga.trainable_weights + self.Gb.trainable_weights))
        self.D_optimizer.apply_gradients(zip(Dgrads, self.Da.trainable_weights + self.Db.trainable_weights))
        return {'l_r':l_r, 'l_cycle':l_cycle, 'g_loss':g_loss, 'd_loss':d_loss}
          

    @tf.function
    def test_step(self, inputs):
        xa, xb = inputs
        
