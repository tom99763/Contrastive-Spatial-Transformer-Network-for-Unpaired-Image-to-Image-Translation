import sys
sys.path.append('./models')
from modules import *
from losses import *
from discriminators import Discriminator
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers


class Encoder(tf.keras.Model):
  def __init__(self, config):
    super().__init__()
    self.act = config['act']
    self.use_bias = config['use_bias']
    self.norm = config['norm']
    self.num_downsampls = config['num_downsamples']
    self.num_resblocks = config['num_resblocks']
    dim = config['base']
    
    self.blocks = tf.keras.Sequential([
        Padding2D(3, pad_type='reflect'),
        ConvBlock(dim, 7, padding='valid', use_bias=self.use_bias, norm_layer=self.norm, activation=self.act),
    ])
    for _ in range(self.num_downsampls):
      dim = dim  * 2
      self.blocks.add(ConvBlock(dim, 3, strides=2, padding='same', use_bias=self.use_bias, norm_layer=self.norm, activation=self.act))
      
    for _ in range(self.num_resblocks):
      self.blocks.add(ResBlock(dim, 3, self.use_bias, self.norm))
    
  def call(self, x):
    return self.blocks(x)


class Decoder(tf.keras.Model):
  def __init__(self, config, opt):
    super().__init__()
    self.act = config['act']
    self.use_bias = config['use_bias']
    self.norm = config['norm']
    self.num_downsampls = config['num_downsamples']
    self.num_resblocks = config['num_resblocks']
    dim = config['base']
    num_channels = opt.num_channels

    self.blocks = tf.keras.Sequential()
      
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


class Generator(tf.keras.Model):
  def __init__(self, config, opt):
    super().__init__()
    self.E=Encoder(config)
    self.D=Decoder(config, opt)
    
  def call(self, x, training=False):
    h, z = self.encode(x)
    if training:
      x = self.decode(h + z)
    else:
      x = self.D(h)
    return x
  
  def encode(self, x):
    h = self.E(x)
    z = tf.random.normal(h.shape)
    return h, z
  
  def decode(self, x):
    return self.D(x)

  
class UNIT(tf.keras.Model):
  def __init__(self, config, opt):
    self.Ga = Generator(config, opt)
    self.Gb = Generator(config, opt)
    self.Da = Discriminator(config)
    self.Db = Discriminator(config)
    self.style_dim = config['style_dim']
    
    self.inst = tfa.layers.InstanceNormalization(scale=False, center=False)
    
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
      za = tf.random.normal((xa.shape[0], 1, 1, self.style_dim))
      zb = tf.random.normal((xb.shape[0], 1, 1, self.style_dim))
      
      
      ### forward
      #encode
      ha, za_prime = self.Ga.encode(xa)
      hb, zb_prime = self.Gb.encode(xb)
      
      #within domain
      xar = self.Ga.decode(ha + za_prime)
      xbr = self.Gb.decode(hb + zb_prime)
      
      #cross domain
      xba = self.Ga.decode(hb + za)
      xab = self.Gb.decode(ha + zb)
      
      #cyclic encode
      hba, zba = self.Ga.encode(xba)
      hab, zab = self.Gb.encode(xab)
      
      #cyclic decode
      xaba = self.Ga.decode(hab + za_prime)
      xbab = self.Gb.decode(hba + zb_prime)
      
      #discrimination
      
      ### compute loss 
      #reconstruction
      l_r = l1_loss(xa, xar) + l1_loss(xb, xbr)
      
      #latent reconstruction
      l_z = l1_loss(za, zba) + l1_loss(zb, zab)
      
      #content reconstruction
      l_h = l1_loss(ha, hab) + l1_loss(hb, hba)
      
      #cyclic 
      l_cycle = l1_loss(xa, xaba) + l1_loss(xb, xbab)
      
      #adversarial loss
  
  
  @tf.function
  def test_step(self, inputs):
    xa, xb = inputs
  







