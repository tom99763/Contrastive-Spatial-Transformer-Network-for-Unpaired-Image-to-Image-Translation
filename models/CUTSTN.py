import tensorflow as tf
import sys

sys.path.append('./models')
from losses import *
from modules import *
from discriminators import *
from tensorflow.keras import initializers, layers


class BilinearSampler(layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        images, theta = inputs
        homogenous_coordinates = self.grid_generator(batch=tf.shape(images)[0])
        return self.interpolate(images, homogenous_coordinates, theta)

    def build(self, shape):
        b, h, w, c = shape[0]
        self.height = h
        self.width = w

    def advance_indexing(self, inputs, x, y):
        shape = tf.shape(inputs)
        batch_size, _, _ = shape[0], shape[1], shape[2]
        batch_idx = tf.range(0, batch_size)
        batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
        b = tf.tile(batch_idx, (1, self.height, self.width))
        indices = tf.stack([b, y, x], 3)
        return tf.gather_nd(inputs, indices)

    def grid_generator(self, batch):
        x = tf.linspace(-1, 1, self.width)
        y = tf.linspace(-1, 1, self.height)

        xx, yy = tf.meshgrid(x, y)
        xx = tf.reshape(xx, (-1,))
        yy = tf.reshape(yy, (-1,))
        homogenous_coordinates = tf.stack([xx, yy, tf.ones_like(xx)])
        homogenous_coordinates = tf.expand_dims(homogenous_coordinates, axis=0)
        homogenous_coordinates = tf.tile(homogenous_coordinates, [batch, 1, 1])
        homogenous_coordinates = tf.cast(homogenous_coordinates, dtype=tf.float32)
        return homogenous_coordinates

    def interpolate(self, images, homogenous_coordinates, theta):
        with tf.name_scope("Transformation"):
            transformed = tf.matmul(theta, homogenous_coordinates)
            transformed = tf.transpose(transformed, perm=[0, 2, 1])
            transformed = tf.reshape(transformed, [-1, self.height, self.width, 2])

            x_transformed = transformed[:, :, :, 0]
            y_transformed = transformed[:, :, :, 1]

            x = ((x_transformed + 1.) * tf.cast(self.width, dtype=tf.float32)) * 0.5
            y = ((y_transformed + 1.) * tf.cast(self.height, dtype=tf.float32)) * 0.5

        with tf.name_scope("VariableCasting"):
            x0 = tf.cast(tf.math.floor(x), dtype=tf.int32)
            x1 = x0 + 1
            y0 = tf.cast(tf.math.floor(y), dtype=tf.int32)
            y1 = y0 + 1

            x0 = tf.clip_by_value(x0, 0, self.width - 1)
            x1 = tf.clip_by_value(x1, 0, self.width - 1)
            y0 = tf.clip_by_value(y0, 0, self.height - 1)
            y1 = tf.clip_by_value(y1, 0, self.height - 1)
            x = tf.clip_by_value(x, 0, tf.cast(self.width, dtype=tf.float32) - 1.0)
            y = tf.clip_by_value(y, 0, tf.cast(self.height, dtype=tf.float32) - 1)

        with tf.name_scope("AdvanceIndexing"):
            Ia = self.advance_indexing(images, x0, y0)
            Ib = self.advance_indexing(images, x0, y1)
            Ic = self.advance_indexing(images, x1, y0)
            Id = self.advance_indexing(images, x1, y1)

        with tf.name_scope("Interpolation"):
            x0 = tf.cast(x0, dtype=tf.float32)
            x1 = tf.cast(x1, dtype=tf.float32)
            y0 = tf.cast(y0, dtype=tf.float32)
            y1 = tf.cast(y1, dtype=tf.float32)

            wa = (x1 - x) * (y1 - y)
            wb = (x1 - x) * (y - y0)
            wc = (x - x0) * (y1 - y)
            wd = (x - x0) * (y - y0)

            wa = tf.expand_dims(wa, axis=3)
            wb = tf.expand_dims(wb, axis=3)
            wc = tf.expand_dims(wc, axis=3)
            wd = tf.expand_dims(wd, axis=3)

        return tf.math.add_n([wa * Ia + wb * Ib + wc * Ic + wd * Id])


class STN(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.act = config['act']
        self.use_bias = config['use_bias']
        self.norm = config['norm']
        self.localizer = self.build_localizer()
        self.sampler = BilinearSampler()

    def call(self, inputs):
        x, z = inputs
        theta = self.localizer(tf.concat([x, z], axis=-1))  # (b, 2, 3)
        x = self.sampler([x, theta])
        return x

    def build_localizer(self):
        dim = self.config['base']
        blocks = tf.keras.Sequential([
            Padding2D(3, pad_type='reflect'),
            ConvBlock(dim, 7, padding='valid', use_bias=self.use_bias, norm_layer=self.norm, activation=self.act),
        ])

        for _ in range(4):
            dim = min(dim * 2, self.config['max_filters'])
            blocks.add(ConvBlock(dim, 3, strides=2, padding='same',
                                 use_bias=self.use_bias, norm_layer=self.norm))
            blocks.add(layers.LeakyReLU(0.2))
        blocks.add(layers.Flatten())
        blocks.add(layers.Dense(self.config['max_filters'], activation=self.act))
        blocks.add(layers.Dense(
            units=6,
            bias_initializer=initializers.constant([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),  # initialize as A = [I, t]
            kernel_initializer='zeros'))
        blocks.add(layers.Reshape((2, 3)))
        return blocks


class Generator(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self.act = config['act']
        self.use_bias = config['use_bias']
        self.norm = config['norm']
        self.num_downsampls = config['num_downsamples']
        self.num_resblocks = config['num_resblocks']
        dim = config['base']

        # build generator
        self.blocks = tf.keras.Sequential([
            layers.Input([None, None, 6]),
            Padding2D(3, pad_type='reflect'),
            ConvBlock(dim, 7, padding='valid', use_bias=self.use_bias, norm_layer=self.norm, activation=self.act),
        ])

        for _ in range(self.num_downsampls):
            dim = dim * 2
            self.blocks.add(ConvBlock(dim, 3, strides=2, padding='same',
                                      use_bias=self.use_bias, norm_layer=self.norm, activation=self.act))

        for _ in range(self.num_resblocks):
            self.blocks.add(ResBlock(dim, 3, self.use_bias, self.norm))

        for _ in range(self.num_downsampls):
            dim = dim / 2
            self.blocks.add(
                ConvTransposeBlock(dim, 3, strides=2, padding='same', use_bias=self.use_bias, norm_layer=self.norm,
                                   activation=self.act))
        self.blocks.add(Padding2D(3, pad_type='reflect'))
        self.blocks.add(ConvBlock(3, 7, padding='valid', activation='tanh'))

        # build spatial transformer
        self.stn = STN(config)

    def call(self, inputs):
        x, z =inputs
        z = z[:, None, None, :]
        z = tf.repeat(tf.repeat(z, x.shape[1], axis=1), x.shape[2], axis=2)
        x = self.wrap(x, z)
        x = self.blocks(tf.concat([x, z], axis=-1))
        return x

    def wrap(self, x, z):
        return self.stn([x, z])


def Encoder(generator, config):
    nce_layers = config['nce_layers']
    outputs = []
    for idx in nce_layers:
        outputs.append(generator.layers[idx].output)
    return tf.keras.Model(inputs=generator.input, outputs=outputs, name='encoder')


class PatchSampler(tf.keras.Model):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.units = config['units']
        self.num_patches = config['num_patches']
        self.l2_norm = layers.Lambda(
            lambda x: x * tf.math.rsqrt(tf.reduce_sum(tf.square(x), axis=-1, keepdims=True) + 1e-10))

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


class CUTSTN(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self.G = Generator(config)
        self.D = Discriminator(config)
        self.E = Encoder(self.G.blocks, config)
        self.F = PatchSampler(config)
        self.config = config

    def compile(self,
                G_optimizer,
                F_optimizer,
                D_optimizer):
        super().compile()
        self.G_optimizer = G_optimizer
        self.F_optimizer = F_optimizer
        self.D_optimizer = D_optimizer
        self.nce_loss_func = PatchNCELoss(self.config['tau'])

    @tf.function
    def train_step(self, inputs):
        la, xb = inputs

        with tf.GradientTape(persistent=True) as tape:
            # synthesize texture
            z = tf.random.normal((la.shape[0], 3))
            xab = self.G([la, z])

            if self.config['use_identity']:
                xbb = self.G([xb, tf.zeros_like(z)])

            # discrimination
            critic_fake = self.D(xab, training=True)
            critic_real = self.D(xb, training=True)

            ###compute losses
            d_loss, g_loss_ = gan_loss(critic_real, critic_fake, self.config['gan_mode'])

            if self.config['use_identity']:
                nce_idt = self.nce_loss_func(xb, xbb, self.E, self.F, tf.zeros_like(z))
            else:
                nce_idt = 0.

            nce_loss = self.nce_loss_func(la, xab, self.E, self.F, z) + nce_idt

            g_loss = g_loss_ + 0.5 * self.config['lambda_nce'] * nce_loss

        G_grads = tape.gradient(g_loss, self.G.trainable_weights)
        D_grads = tape.gradient(d_loss, self.D.trainable_weights)
        F_grads = tape.gradient(nce_loss, self.F.trainable_weights)

        self.G_optimizer.apply_gradients(zip(G_grads, self.G.trainable_weights))
        self.D_optimizer.apply_gradients(zip(D_grads, self.D.trainable_weights))
        self.F_optimizer.apply_gradients(zip(F_grads, self.F.trainable_weights))

        del tape

        return {'g_loss': g_loss_, 'd_loss': d_loss, 'nce': nce_loss}

    @tf.function
    def test_step(self, inputs):
        la, xb = inputs
        z = tf.random.normal((la.shape[0], 3))
        xab = self.G([la, z])

        if self.config['use_identity']:
            xbb = self.G([xb, tf.zeros_like(z)])
            nce_idt = self.nce_loss_func(xb, xbb, self.E, self.F, tf.zeros_like(z))
        else:
            nce_idt = 0.

        nce_loss = self.nce_loss_func(la, xab, self.E, self.F, z) + nce_idt
        return {'nce': nce_loss}

