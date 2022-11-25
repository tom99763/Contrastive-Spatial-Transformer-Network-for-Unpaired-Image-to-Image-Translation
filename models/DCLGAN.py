import tensorflow as tf
import sys
sys.path.append('./models')
from losses import *
from modules import *
from discriminators import *
from tensorflow.keras import layers

class Generator(tf.keras.Model):
  def __init__(self, config, opt):
    super().__init__()
    self.act = config['act']
    self.use_bias = config['use_bias']
    self.norm = config['norm']
    self.num_downsampls = config['num_downsamples']
    self.num_resblocks = config['num_resblocks']
    self.num_channels = opt.num_channels
    dim = config['base']
    
    self.blocks = tf.keras.Sequential([
        layers.Input([None, None, 3]),
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
    self.blocks.add(ConvBlock(self.num_channels, 7, padding='valid', activation='tanh'))
    
  def call(self, x):
    return self.blocks(x)
  
  
def Encoder(generator, config):
    nce_layers = config['nce_layers']
    outputs = []
    for idx in nce_layers:
        outputs.append(generator.layers[idx].output)
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
      
class DCLGAN(tf.keras.Model):
  def __init__(self, config, opt):
    super().__init__()
    self.Ga = Generator(config, opt)
    self.Gb = Generator(config, opt)
    self.Fa = PatchSampleMLP(config)
    self.Fb = PatchSampleMLP(config)
    self.Da = Discriminator(config)
    self.Db = Discriminator(config)
    
    self.Ea = Encoder(self.Ga.blocks, config)
    self.Eb = Encoder(self.Gb.blocks, config)
    
    self.gan_mode = config['gan_mode']
    self.use_identity = config['use_identity']
    self.lambda_nce = config['lambda_nce']
    self.tau = config['tau']
    
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
    self.nce_loss_func = PatchNCELoss(self.tau)
  
  @tf.function
  def train_step(self, inputs):
    xa, xb = inputs
    
    with tf.GradientTape(persistent=True) as tape:
      xab = self.Gb(xa)
      xba = self.Ga(xb)
      
      if self.use_identity:
        xaa = self.Ga(xa)
        xbb = self.Gb(xb)
      
      #discrimination
      critic_real_a = self.Da(xa)
      critic_fake_a = self.Da(xba)
      critic_real_b = self.Db(xb)
      critic_fake_b = self.Da(xab)
      
      ###compute loss
      #infonce
      l_nce_a = self.nce_loss_func(xb , xba, self.Ea, self.Fa)
      l_nce_b = self.nce_loss_func(xa , xab, self.Eb, self.Fb)
      
      #identity
      if use_identity:
        l_idt_a = l1_loss(xa, xaa)
        l_idt_b = l1_loss(xb, xbb)
      else:
        l_idt_a = 0.
        l_idt_b = 0.
      
      #adversarial 
      da_loss, ga_loss = gan_loss(critic_real_a, critic_fake_a, self.config['gan_mode'])
      db_loss, gb_loss = gan_loss(critic_real_b, critic_fake_b, self.config['gan_mode'])
      
      l_ga = ga_loss + l_idt_a + l_nce_a
      l_gb = gb_loss + l_idt_b + l_nce_b
      l_da = da_loss
      l_db = db_loss

    Gagrads = tape.gradient(l_ga, self.Ga.trainable_weights)
    Gbgrads = tape.gradient(l_gb, self.Gb.trainable_weights)
    Dagrads = tape.gradient(l_da, self.Da.trainable_weights)
    Dbgrads = tape.gradient(l_db, self.Db.trainable_weights)

    self.Ga_optimizer.apply_gradients(zip(Gagrads, self.Ga.trainable_weights))
    self.Gb_optimizer.apply_gradients(zip(Gbgrads, self.Gb.trainable_weights))
    self.Da_optimizer.apply_gradients(zip(Dagrads, self.Da.trainable_weights))
    self.Db_optimizer.apply_gradients(zip(Dbgrads, self.Db.trainable_weights))
    
    return {'nce': 0.5 * (l_nce_a + l_nce_b),'idt': 0.5 * (l_idt_a + l_idt_b),
            'g_loss': 0.5 * (ga_loss + gb_loss), 'd_loss': 0.5 * (da_loss + db_loss)}
      
  
  @tf.function
  def test_step(self, inputs):
    pass
      
