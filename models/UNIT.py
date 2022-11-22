import sys
sys.path.append('./models')
from modules import *
from losses import *
from discriminators import Discriminator
import tensorflow as tf
from tensorflow.keras import layers


class VAEGen(nn.Module):
    # VAE architecture
    def __init__(self, input_dim, params):
        super(VAEGen, self).__init__()
        dim = params['dim']
        n_downsample = params['n_downsample']
        n_res = params['n_res']
        activ = params['activ']
        pad_type = params['pad_type']

        # content encoder
        self.enc = ContentEncoder(n_downsample, n_res, input_dim, dim, 'in', activ, pad_type=pad_type)
        self.dec = Decoder(n_downsample, n_res, self.enc.output_dim, input_dim, res_norm='in', activ=activ, pad_type=pad_type)

    def forward(self, images):
        # This is a reduced VAE implementation where we assume the outputs are multivariate Gaussian distribution with mean = hiddens and std_dev = all ones.
        hiddens = self.encode(images)
        if self.training == True:
            noise = Variable(torch.randn(hiddens.size()).cuda(hiddens.data.get_device()))
            images_recon = self.decode(hiddens + noise)
        else:
            images_recon = self.decode(hiddens)
        return images_recon, hiddens

    def encode(self, images):
        hiddens = self.enc(images)
        noise = Variable(torch.randn(hiddens.size()).cuda(hiddens.data.get_device()))
        return hiddens, noise

    def decode(self, hiddens):
        images = self.dec(hiddens)
        return images


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
    num_channels = opt.num_channels

    self.blocks = tf.keras.Sequential()
      
    for _ in range(self.num_resblocks):
      self.blocks.add(ResBlock(dim, 3, self.use_bias, self.norm))
      
    for _ in range(self.num_downsampls):
      dim  = dim / 2
      self.blocks.add(ConvTransposeBlock(dim, 3, strides=2, padding='same',
                                         use_bias=self.use_bias, norm_layer=self.norm, activation=self.act))
    self.blocks.add(Padding2D(3, pad_type='reflect'))
    self.blocks.add(ConvBlock(3, 7, padding='valid', activation='tanh'))
    
  def call(self, x):
    return self.blocks(x)


class Generator(tf.keras.Model):
  def __init__(self, config, opt):
    super().__init__()
    self.E=Encoder(config)
    self.D=Decoder(config, opt)
    
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
  







