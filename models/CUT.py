import tensorflow as tf
from losses import *
from modules import *
from discriminators import *

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
  
  
class PatchSampleMLP(tf.keras.Model):
    def __init__(self, config **kwargs):
        super(PatchSampleMLP, self).__init__(**kwargs)
        self.units = config['units']
        self.num_patches = config['num_patches']
        self.l2_norm = Lambda(lambda x: x * tf.math.rsqrt(tf.reduce_sum(tf.square(x), axis=-1, keepdims=True) + 10-10))

    def build(self, input_shape):
        initializer = tf.random_normal_initializer(0., 0.02)
        feats_shape = input_shape
        for feat_id in range(len(feats_shape)):
            mlp = tf.keras.models.Sequential([
                    Dense(self.units, activation="relu", kernel_initializer=initializer),
                    Dense(self.units, kernel_initializer=initializer),
                ])
            setattr(self, f'mlp_{feat_id}', mlp)

    def call(self, inputs, patch_ids=None, training=None):
        feats = inputs
        samples = []
        ids = []
        for feat_id, feat in enumerate(feats):
            B, H, W, C = feat.shape
            feat_reshape = tf.reshape(feat, [B, -1, C])
            if patch_ids is not None:
                patch_id = patch_ids[feat_id]
            else:
                patch_id = tf.random.shuffle(tf.range(H * W))[:min(self.num_patches, H * W)]
            x_sample = tf.reshape(tf.gather(feat_reshape, patch_id, axis=1), [-1, C])
            mlp = getattr(self, f'mlp_{feat_id}')
            x_sample = mlp(x_sample)
            x_sample = self.l2_norm(x_sample)
            samples.append(x_sample)
            ids.append(patch_id)
        return samples, ids

  
class CVQAE(tf.keras.Model):
  def __init__(self):
    super().__init__()
    
    self.G = Generator(config)
    self.Disc = Discriminator(config)
    self.PatchSampler = PatchSampleMLP(config)
  
  @tf.function
  def train_step(self, inputs):
    return 
  
  @tf.function
  def test_step(self, inputs):
    return 
