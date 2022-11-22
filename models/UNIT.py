import sys
sys.path.append('./models')
from modules import *
from losses import *
from discriminators import Discriminator
import tensorflow as tf
from tensorflow.keras import layers


class Encoder(tf.keras.Model):
  pass


class Decoder(tf.keras.Model):
  pass


class Generator(tf.keras.Model):
  def __init__(self, config, opt):
    super().__init__()
    
  def call(self, x):
    return 
  
class UNIT(tf.keras.Model):
  def __init__(self, config, opt):
    self.Ga = Generator(config, opt)
    self.Gb = Generator(config, opt)
    self.Da = Discriminator(config)
    self.Db = Discriminator(config)
    
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
  
  @tf.function
  def test_step(self, inputs):
    xa, xb = inputs
  







