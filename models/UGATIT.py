import sys

sys.path.append('./models')
from modules import *
from losses import *
from discriminators import Discriminator
import tensorflow as tf
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


class Decoder:
  pass


class Generator:
  pass


class UGATIT:
  pass
