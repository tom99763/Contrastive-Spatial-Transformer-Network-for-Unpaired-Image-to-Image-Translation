from modules import *
import tensorflow as tf

class UNET(tf.keras.Model):
  def __init__(self, config):
    super().__init__()
    
  def call(self, x):
    pass

class Generator(tf.keras.Model):
  def __init__(self, config):
    super().__init__()
    self.encoder = self.build_encoder()
    
  def call(self, x):
    pass
  
  def build_encoder(self):
    nce_layers = self.config['nce_layers']
    outputs = []
    for idx in nce_layers:
      outputs.append(self.layers[idx].output)
    return tf.keras.Model(inputs=self.input, outputs=outputs)


class PatchSampler(tf.keras.Model):
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

      
class STN(tf.keras.Model):
  def __init__(self, config):
    super().__init__()

  def call(self, x):
    pass


class CUTSTN(tf.keras.Model):
  def __init__(self, config):
    super().__init__()
    self.landmarker = UNET(config)
    self.synthesizer = Generator(config)
    self.discriminator = Discriminator(config)
    self.patch_sampler = PatchSampler(config)
    self.stn =STN(config)
  
  @tf.function
  def train_step(self, inputs):
    pass
  
  @tf.function
  def test_step(self, inputs):
    pass
  
  
  
