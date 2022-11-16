from modules import *
import tensorflow as tf


class Discriminator(tf.keras.Model):
  def __init__(self, config):
    super().__init__()
    disc_type = config['disc_type']
    
    if disc_type == 'classic':
      self.disc =None
    
    elif disc_type == 'patch':
      self.disc = Patch_Discriminator(config)
      
    elif disc_type == 'perceptual':
      self.disc = Perceptual_Discriminator(config)
    
    elif disc_type == 'Multi_scale':
      self.disc = None
    
  def call(self, x):
    return self.disc(x)


class Patch_Discriminator(tf.keras.Model):
  def __init__(self, config):
    super().__init__()
    self.act = config['act']
    self.use_bias = config['use_bias']
    self.norm = config['norm']
    self.num_downsampls = config['num_downsamples']
    self.num_resblocks = config['num_resblocks']
    dim = config['base']
    
    self.blocks = tf.keras.Sequential([
      ConvBlock(dim, 4, strides=2, padding='same', use_bias=self.use_bias, norm_layer=self.norm, activation=tf.nn.leaky_relu)
    ])
    
    for _ in range(self.num_downsampls):
      dim = dim * 2
      self.blocks.add(ConvBlock(dim, 4, strides=2, padding='same', use_bias=self.use_bias, norm_layer=self.norm, activation=tf.nn.leaky_relu))
      
    self.blocks.add(Padding2D(1, pad_type='constant'))
    self.blocks.add(ConvBlock(512, 4, padding='valid', use_bias=self.use_bias, norm_layer=self.norm, activation=tf.nn.leaky_relu))
    self.blocks.add(Padding2D(1, pad_type='constant'))
    self.blocks.add(ConvBlock(1, 4, padding='valid'))
    
  def call(self, x):
    return self.blocks(x)
  
  
class Perceptual_Discriminator(tf.keras.Model):
  def __init__(self, config):
    super().__init__()
    dim = config['base']
    
  def call(self, x):
    return 
  
  
  
