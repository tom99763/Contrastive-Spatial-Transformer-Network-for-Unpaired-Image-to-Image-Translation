import sys
sys.path.append('./models')
from modules import *
from losses import *
from tensorflow.keras.applications.vgg16 import VGG16

class CoordPredictor(tf.keras.Model):
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
      
    for _ in range(self.num_downsampls):
      dim  = dim / 2
      self.blocks.add(ConvTransposeBlock(dim, 3, strides=2, padding='same',
                                         use_bias=self.use_bias, norm_layer=self.norm, activation=self.act))
    self.blocks.add(Padding2D(3, pad_type='reflect'))
    self.blocks.add(ConvBlock(2, 7, padding='valid', activation='tanh'))
    
  def call(self, inputs):
    x = tf.concat(inputs, axis=-1)
    return self.blocks(x)
 

class Encoder(tf.keras.Model):
  def __init__(self, config):
    super().__init__()
    self.nce_layers=config['nce_layers']
    self.vgg = self.build_vgg()
    
  def call(self, x):
    return self.vgg(x)
  
  def build_vgg(self):
    vgg = VGG16(include_top=False)
    vgg.trainable=False
    outputs = [vgg.layers[idx].output for idx in self.nce_layers]
    return tf.keras.Model(inputs=vgg.input, outputs=outputs)
  

class GridSampler:
  pass



class PatchSampler:
  pass



class InfoMatch(tf.keras.Model):
  def __init__(self, config):
    super().__init__()
    self.CP
    self.E
    self.F
    self.GS
    
  @tf.function
  def train_step(self, inputs):
    xa, xb = inputs
    
    with tf.GradientTape(persistent=True) as tape:
      coord = self.CP([xa, xb])
      xa_wrapped = self.GS([xa, coord])
      l_info = self.PatchInfoNCE(xb, xa_wrapped, self.E, self.F)
      
    grads = tape.gradient(l_info, self.CP.trainable_weights + self.F.trainable_weights)
    self.optimizer.apply_gradients(zip(grads, self.G.trainable_weights + self.F.trainable_weights))
    return {'infonce':l_info}
      
  
  @tf.function
  def test_step(self, inputs):
    xa, xb = inputs
    coord = self.CP([xa, xb])
    xa_wrapped = self.GS([xa, coord])
    l_info = self.PatchInfoNCE(xb, xa_wrapped, self.E, self.F)
    return {'infonce':l_info}
