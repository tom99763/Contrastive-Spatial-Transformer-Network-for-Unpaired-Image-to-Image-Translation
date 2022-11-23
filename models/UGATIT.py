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
    self.num_mlps = config['num_mlps']
    dim = config['max_filters']  
    
    #cam 
    self.gap=layers.GloablAveragePooling2D()
    self.gmp=layers.GlobalMaxPool2D()
    self.w = tf.Variable(tf.random.normal([dim, 1]), trainable=True)
    self.fuse = ConvBlock(dim, 1, padding='valid', 
                          use_bias=self.use_bias, activation=self.act)
    self.wMap = tf.keras.Sequential([layers.Dense(dim, activation = self.act) for _ in range(self.num_mlps-1)])
    self.wMap.add(layers.Dense(dim))
    
    #upsample
    self.resblocks=[Resblock(dim ,3, self.use_bias, 'adaptive_layer_instance') for _ in range(self.num_resblocks)]
    
    self.blocks = tf.keras.Sequential()
    for _ in range(self.num_downsampls):
      dim  = dim / 2
      self.blocks.add(ConvTransposeBlock(dim, 3, strides=2, padding='same',
                                         use_bias=self.use_bias, norm_layer='layer_instance', activation=self.act))
    self.blocks.add(Padding2D(3, pad_type='reflect'))
    self.blocks.add(ConvBlock(opt.num_channels, 7, padding='valid', activation='tanh'))
  
  def call(self, x):
    w, cam_logits = self.cam(x)
    for block in self.resblocks:
      x = self.block([x, w])
    x = self.blocks(x)
    return x, cam_logits
  
  def cam(self):
    #global average pooling
    cam_gap = self.gap(x)
    cam_gap_logits, cam_gap_weights = self.cMap(cam_gap)
    x_gap = x * cam_gap_weights #(b, h, w, c)
    
    #global maximam pooling
    cam_gmp = self.mlp(x)
    cam_gmp_logits, cam_gmp_weights = self.cMap(cam_gmp)
    x_gmp = x * cam_gmp_weights #(b, h, w, c)
    
    #fusion
    cam_logits = tf.concat([cam_gap_logits, cam_gmp_logits], axis=-1) #(b, hw, 2)
    x = tf.concat([x_gap, x_gmp], axis=-1) 
    x = self.fuse(x) #(b, h, w, c)
    
    #wmap
    x = self.gap(x)
    w = self.wMap(x)
    return w, cam_logits #(b, d), (b, hw, 2)
  
  def cMap(self, x):
    b, h, w, c = x.shape
    x= layers.Flatten()(x) #(b, hw, c)
    x = x @ self.w #(b, hw, 1)
    w = tf.gather(tf.transpose(self.w), 0) #(dim, )
    return x, w
    

class Generator(tf.keras.Model):
  def __init__(self, config, opt):
    super().__init__()
    self.E=Encoder(config)
    self.D=Decoder(config, opt)
  def call(self, x):
    x = self.E(x)
    x ,cam_logits = self.D(x)
    return x, cam_logits


class UGATIT(tf.keras.Model):
  def __init__(self, config, opt):
    super().__init__()
    
    self.Ga = Generator(config, opt)
    self.Gb = Generator(config, opt)
    
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
      
      
    def test_step(self, inputs):
      xa, xb = inputs
      
      
    
    
    
    
    
