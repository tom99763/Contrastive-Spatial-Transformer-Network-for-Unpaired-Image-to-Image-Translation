import sys
sys.path.append('./models')
from modules import *
from losses import *
from tensorflow.keras.applications.vgg16 import VGG16

class CoordPredictor(tf.keras.Model):
  def __init__(self, conifg):
    super().__init__()
    
  def call(self, x):
    return
  
def Encoder():
  pass

class GridSampler:
  pass

class PatchSampler:
  pass

class InfoMatchNet(tf.keras.Model):
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
