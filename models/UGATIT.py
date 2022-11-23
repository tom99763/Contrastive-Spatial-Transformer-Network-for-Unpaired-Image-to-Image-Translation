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


class Decoder(tf.keras.Model):
  def __init__(self, config, opt):
    super().__init__()
    self.act = config['act']
    self.use_bias = config['use_bias']
    self.norm = config['norm']
    self.num_downsampls = config['num_downsamples']
    self.num_resblocks = config['num_resblocks']
    dim = config['base']    
    
    #cam 
    self.gap=layers.GloablAveragePooling2D()
    self.gmp=layers.GlobalMaxPooling2D()
    self.w = tf.Variable(tf.random.normal([self.config['max_filters'], 1]), trainable=True)
    self.fuse = ConvBlock(config['max_filters'], 1, padding='valid', 
                          use_bias=self.use_bias, activation=self.act)
    
    #upsample
    self.blocks=[AdaLINResBlock() for _ in range(self.num_resblocks)]
    
    for _ in range(self.num_downsampls):
      dim  = dim / 2
      self.blocks.add(ConvTransposeBlock(dim, 3, strides=2, padding='same',
                                         use_bias=self.use_bias, norm_layer='layer_instance', activation=self.act))
    self.blocks.add(Padding2D(3, pad_type='reflect'))
    self.blocks.add(ConvBlock(3, 7, padding='valid', activation='tanh'))
  
  def call(self, x):
    w = self.cam(x)
    for block in self.blocks:
      x = self.block([x, w])
    return x
  
  def cam(self):
    #global average pooling
    cam_gap = self.gap(x)
    cam_gap_logit, cam_gap_weight = self.cMap(cam_gap)
    x_gap = x * cam_gap_weight
    
    #global maximam pooling
    cam_gmp = self.mlp(x)
    cam_gmp_logit, cam_gmp_weight = self.cMap(cam_gmp)
    x_gmp = x * cam_gmp_weight
    
    #fusion
    cam_logit = tf.concat([cam_gap_logits, cam_gmp_logits], axis=-1) #(b, hw, 2)
    x = tf.concat([x_gap, x_gmp], axis=-1)
    x = self.fuse(x)
    
    #wmap
    w = self.wMap(x)
    return w, cam_logits 
  
  def cMap(self, x):
    b, h, w, c = x.shape
    x= layers.Flatten()(x) #(b, hw, c)
    x = x @ self.w #(b, hw, 1)
    w = tf.gather(tf.transpose(self.w), 0)
    return x, w
    


class Generator:
  pass


class UGATIT:
  pass
