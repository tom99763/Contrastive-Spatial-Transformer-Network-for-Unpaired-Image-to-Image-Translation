import tensorflow as tf
from losses import *
from modules import *

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
      Padding2D(3, pad_type='reflect'),
      ConvBlock(dim, 7, padding='valid', use_bias=self.use_bias, norm_layer=self.norm, activation=self.act),
    ])
    
    for _ in range(self.num_downsampls):
      dim * = 2
      self.blocks.add(ConvBlock(dim, 3, strides=2, padding='same', use_bias=self.use_bias, norm_layer=self.norm, activation=self.act))
      
    for _ in range(self.num_resblocks):
      self.blocks.add(ResBlock(dim, 3, self.use_bias, self.norm_layer))
      
    for _ in range(self.num_downsampls):
      dim / = 2
      self.blocks.add(ConvTransposeBlock(dim, 3, strides=2, padding='same', use_bias=self.use_bias, norm_layer=self.norm, activation=self.act))
    self.blocks.add(Padding2D(3, pad_type='reflect'))
    self.blocks.add(ConvBlock(3, 7, padding='valid', activation='tanh'))
    
  def call(self, x):
    return self.blocks(x)

  
class CVQAE(tf.keras.Model):
  def __init__(self):
    super().__init__()
    
    self.G = Generator(config)
    self.Disc = Discriminator(config)
    self.mlp = None
  
  @tf.function
  def train_step(self, inputs):
    return 
  
  @tf.function
  def test_step(self, inputs):
    return 
