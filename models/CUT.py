import tensorflow as tf
from losses import *
from modules import *
from discriminators import *
from tensorflow.keras import layers

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
      dim = dim  * 2
      self.blocks.add(ConvBlock(dim, 3, strides=2, padding='same', use_bias=self.use_bias, norm_layer=self.norm, activation=self.act))
      
    for _ in range(self.num_resblocks):
      self.blocks.add(ResBlock(dim, 3, self.use_bias, self.norm_layer))
      
    for _ in range(self.num_downsampls):
      dim  = dim / 2
      self.blocks.add(ConvTransposeBlock(dim, 3, strides=2, padding='same', use_bias=self.use_bias, norm_layer=self.norm, activation=self.act))
    self.blocks.add(Padding2D(3, pad_type='reflect'))
    self.blocks.add(ConvBlock(3, 7, padding='valid', activation='tanh'))
    
  def call(self, x):
    return self.blocks(x)
  
  
def Encoder(generator, config):
  nce_layers = config['nce_layers']
  assert max(nce_layers) <= len(generator.layers) and min(nce_layers) >= 0
  outputs = [generator.get_layer(index=idx).output for idx in nce_layers]
  return tf.keras.Model(inputs=generator.input, outputs=outputs, name='encoder')

  
class PatchSampleMLP(tf.keras.Model):
    def __init__(self, config, **kwargs):
        super(PatchSampleMLP, self).__init__(**kwargs)
        self.units = config['units']
        self.num_patches = config['num_patches']
        self.l2_norm = layers.Lambda(lambda x: x * tf.math.rsqrt(tf.reduce_sum(tf.square(x), axis=-1, keepdims=True) + 1e-10))

    def build(self, input_shape):
        initializer = tf.random_normal_initializer(0., 0.02)
        feats_shape = input_shape
        for feat_id in range(len(feats_shape)):
            mlp = tf.keras.models.Sequential([
                    layers.Dense(self.units, activation="relu", kernel_initializer=initializer),
                    layers.Dense(self.units, kernel_initializer=initializer),
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

  
class CUT(tf.keras.Model):
  def __init__(self, config):
    super().__init__()
    
    self.G = Generator(config)
    self.D = Discriminator(config)
    self.E = Encoder(self.G, config)
    self.F = PatchSampleMLP(config)
    
    self.gan_mode = config['gan_mode']
    self.use_identity = config['use_identity']
    self.lambda_nce = config['lambda_nce']
    self.tau = config['tau']
    
  def compile(self,
              G_optimizer,
              F_optimizer,
              D_optimizer):
      super(CUT, self).compile()
      self.G_optimizer = G_optimizer
      self.F_optimizer = F_optimizer
      self.D_optimizer = D_optimizer
      self.nce_loss_func = PatchNCELoss(self.tau)
  
  @tf.function
  def train_step(self, inputs):
    source, target = inputs
    x = tf.concat([source, target], axis=0) if self.use_identity else source
    
    with tf.GradientTape(persistent=True) as tape:
      y = self.G(x, training=True)
      x2y = y[:source.shape[0]]
      
      if self.use_identity:
        y_idt = x2y[source.shape[0]:]
      
      critic_fake = self.D(x2y, training=True)
      critic_real = self.D(y, training=True)
      
      ###compute loss
      d_loss, g_loss_ = gan_loss(critic_real, critic_fake, self.gan_mode)
      
      nce_loss = self.nce_loss_func(source, x2y, self.E, self.F)
      
      if self.use_identity:
        nce_idt_loss = self.nce_loss_func(target, y_idt, self.E, self.F)
        nce_loss = (nce_loss + nce_idt_loss) * 0.5
        
      g_loss = g_loss_ + self.lambda_nce * nce_loss
      
    G_grads = tape.gradient(g_loss, self.G.trainable_weights)
    D_grads = tape.gradient(d_loss, self.D.trainable_weights)
    F_grads = tape.gradient(nce_loss, self.F.trainable_weights)
    
    self.G_optimizer.apply_gradients(zip(G_grads, self.G.trainable_weights))
    self.D_optimizer.apply_gradients(zip(D_grads, self.D.trainable_weights))
    self.F_optimizer.apply_gradients(zip(F_grads, self.F.trainable_weights))
      
    del tape
    
    return {'g_loss': g_loss_, 'd_loss':d_loss, 'nce': nce_loss}
  
  @tf.function
  def test_step(self, inputs):
    source, target = inputs
    x = tf.concat([source, target], axis=0) if self.use_identity else source
    
    y = self.G(x, training=True)
    x2y = y[:source.shape[0]]
      
    if self.use_identity:
      y_idt = x2y[source.shape[0]:]

    ###compute loss
    nce_loss = self.nce_loss_func(source, x2y, self.E, self.F)
      
    if self.use_identity:
      nce_idt_loss = self.nce_loss_func(target, y_idt, self.E, self.F)
      nce_loss = (nce_loss + nce_idt_loss) * 0.5

    return {'nce': nce_loss}
