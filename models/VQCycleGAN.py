from modules import *
from losses import *
import tensorflow as tf
from tensorflow.keras import layers
from discriminators import *


class VectorQuantizer(layers.Layer):
    def __init__(self, config):
        super().__init__()
        self.n_embs=config['n_embs']
        self.beta=config['beta']
        
    def build(self,shape):
        emb_dim=shape[-1]
        init=tf.random_uniform_initializer()
        self.embs=tf.Variable(
            initial_value=init(shape=[self.n_embs,emb_dim],dtype='float32'),
            trainable=True,
            name='embeddings'
        )
        
    def call(self,ze):
        b,h,w,c=ze.shape
        z=tf.reshape(ze,[-1,c])
        idx=self.get_code_idx(z)
        zq=tf.gather(self.embs,idx,axis=0)
        zq=tf.reshape(zq,[b,h,w,c])
        
        #loss
        commitment_loss=self.beta*tf.reduce_mean((tf.stop_gradient(zq)-ze)**2)
        coodbook_loss=tf.reduce_mean((zq-tf.stop_gradient(ze))**2)
        self.add_loss(commitment_loss+coodbook_loss)
        
        #straight-through estimator to update the encoder
        zq=ze+tf.stop_gradient(zq-ze)
        return zq
    
    def get_code_idx(self,z):
        #distance metric
        similarity=z@tf.transpose(self.embs) #[bhw,n_embs]
        distances = tf.reduce_sum(z ** 2, axis=-1,keepdims=True)\
                 + tf.expand_dims(tf.reduce_sum(self.embs** 2, axis=-1),axis=0)\
                 - 2 * similarity
        #align codes
        idx=tf.argmin(distances,axis=-1) #[bhw]
        return idx
  

class Encoder(tf.keras.Model):
  pass


class Decoder(tf.keras.Model):
  pass


class Generator(tf.keras.Model):
  def __init__(self, config):
    super().__init__()
    self.E = Encoder(config)
    self.D = Decoder(config)
    self.VQ = VectorQuantizer(config)
    
  def call(self, x):
    x = self.E(x)
    x = self.VQ(x)
    x = self.D(x)
    return x


class VQCUT(tf.keras.Model):
  def __init__(self, config):
    self.Ga = Generator(config)
    self.Gb = Generator(config)
    self.Da = Discriminator(config)
    self.Db = Discrminator(config)
    
  @tf.function
  def train_step(self, inputs):
    pass
    
    
