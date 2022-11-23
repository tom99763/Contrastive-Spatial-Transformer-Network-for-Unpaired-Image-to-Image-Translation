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
    self.gap = layer.GlobalAveragePooling2D()
    self.gmp = layers.GlobalMaxPool2D()
    self.gap_fc = layers.Dense(1, use_bias=False) 
    self.gmp_fc = layers.Dense(1, use_bias=False)
    self.fuse =  ConvBlock(dim, 1, activation=self.act)
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
  
  def cam(self, x):
    #average
    gap = self.gap(x)
    gap_logits = self.gap_fc(gap)
    gap_weights = self.gap_fc.trainable_weights[0][:, 0] 
    x_gap = x * gap_weights
    
    #maximum
    gmp = self.gap(x)
    gmp_logits = self.gmp_fc(gmp)
    gmp_weights = self.gmp_fc.trainable_weights[0][:, 0]
    x_gmp = x * gmp_weights
    
    #output
    cam_logits = tf.concat([gap_logits, gmp_logits], axis=-1) #(b, )
    x = tf.concat([x_gap, x_gmp], axis=-1) #(b, h, w, 2c)
    x = self.fuse(x) #(b, h, w, c)
    w = self.wMap(x)
    return w, cam_logits

    
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
    self.Da = Discriminator(config)
    self.Db = Discriminator(config)
    self.config = config
    
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
      
      with tf.GradientTape(persistent=True) as tape:
        ###forward
        #identity
        xaa, cam_logits_aa = self.Ga(xa)
        xbb, cam_logits_bb = self.Ga(xb)
        
        #translation
        xab, cam_logits_ab = self.Gb(xa)
        xba, cam_logits_ba = self.Ga(xb)
        
        #cyclic
        xaba, _ = self.Ga(xab)
        xbab, _ = self.Gb(xba)
        
        #discrimination
        critic_real_a, critic_real_cam_logits_a = self.Da(xa)
        critic_real_b, critic_real_cam_logits_b = self.Db(xb)
        critic_fake_a, critic_fake_cam_logits_a = self.Da(xba)
        critic_fake_b, critic_fake_cam_logits_b = self.Db(xab)
        
        ###compute loss
        #identity
        l_ra = l1_loss(xa, xaa)
        l_rb = l1_loss(xb, xbb)
        
        #cyclic
        l_cycle = l1_loss(xa, xaba) + l1_loss(xb, xbab)
        
        #cam generator
        l_ga_cam = bc(tf.ones_like(cam_logits_ba), cam_logits_ba) +\
                   bc(tf.zeros_like(cam_logits_aa), cam_logits_aa)
        
        l_gb_cam = bc(tf.ones_like(cam_logits_ab), cam_logits_ab) +\
                   bc(tf.zeros_like(cam_logits_bb), cam_logits_bb)
        
        #cam discriminator
        l_da_cam = bc(tf.ones_like(critic_real_cam_logits_a), critic_real_cam_logits_a) + \
                   bc(tf.zeros_like(critic_fake_cam_logits_a), critic_fake_cam_logits_a)
        l_db_cam = bc(tf.ones_like(critic_real_cam_logits_b), critic_real_cam_logits_b) + \
                   bc(tf.zeros_like(critic_fake_cam_logits_b), critic_fake_cam_logits_b)
        
        l_dga_cam = bc(tf.ones_like(critic_fake_cam_logits_a), critic_fake_cam_logits_a)
        l_dgb_cam = bc(tf.ones_like(critic_fake_cam_logits_b), critic_fake_cam_logits_b)
        
        #adversarial loss
        da_loss, ga_loss = gan_loss(critic_real_a, critic_fake_a, self.config['gan_mode'])
        db_loss, gb_loss = gan_loss(critic_real_b, critic_fake_b, self.config['gan_mode'])
        
        #total loss
        l_ga = l_ra + l_cycle + l_ga_cam + l_dga_cam + ga_loss
        l_gb = l_rb + l_cycle + l_gb_cam + l_dgb_cam + gb_loss
        l_da = l_da_cam + da_loss
        l_db = l_db_cam + db_loss

      Gagrads = tape.gradient(ga_loss, self.Ga.trainable_weights)
      Gbgrads = tape.gradient(gb_loss, self.Gb.trainable_weights)
      Dagrads = tape.gradient(da_loss, self.Da.trainable_weights)
      Dbgrads = tape.gradient(db_loss, self.Db.trainable_weights)

      self.Ga_optimizer.apply_gradients(zip(Gagrads, self.Ga.trainable_weights))
      self.Gb_optimizer.apply_gradients(zip(Gbgrads, self.Gb.trainable_weights))
      self.Da_optimizer.apply_gradients(zip(Dagrads, self.Da.trainable_weights))
      self.Db_optimizer.apply_gradients(zip(Dbgrads, self.Db.trainable_weights))
      
      return {'l_r':0.5 *(l_ra + l_rb), 'l_cycle':l_cycle, 'l_g_cam': 0.5 * (l_ga_cam + l_gb_cam),
              'l_d_cam' : 0.5 * (l_da_cam + l_db_cam), 'l_dg':0.5 * (l_dga_cam + l_dgb_cam), 
              'g_loss':0.5 * (l_ga + l_gb), 'd_loss': 0.5 * (l_da + l_db)
             }
      
      
    def test_step(self, inputs):
      xa, xb = inputs
      
      
    
    
    
    
    
