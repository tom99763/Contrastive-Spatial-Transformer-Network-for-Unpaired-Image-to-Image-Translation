import sys

sys.path.append('./models')
from modules import *
from losses import *
from tensorflow.keras.applications.vgg16 import VGG16
import tensorflow as tf
from tensorflow.keras import layers


class CoordPredictor(tf.keras.Model):
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
            dim = dim * 2
            self.blocks.add(ConvBlock(dim, 3, strides=2, padding='same', use_bias=self.use_bias, norm_layer=self.norm,
                                      activation=self.act))

        for _ in range(self.num_resblocks):
            self.blocks.add(ResBlock(dim, 3, self.use_bias, self.norm))

        for _ in range(self.num_downsampls):
            dim = dim / 2
            self.blocks.add(ConvTransposeBlock(dim, 3, strides=2, padding='same',
                                               use_bias=self.use_bias, norm_layer=self.norm, activation=self.act))
        self.blocks.add(Padding2D(3, pad_type='reflect'))
        self.blocks.add(ConvBlock(2, 7, padding='valid', activation='tanh'))

    def call(self, inputs):
        xa, xb = inputs
        x = tf.concat([xa, xb], axis=-1)
        grids_shift = self.blocks(x)
        grids_shift = grids_shift / 10.

        grids = affine_grid_generator(x.shape[1], x.shape[2], x.shape[0]) +\
                tf.transpose(grids_shift, perm=[0, 3, 1, 2])
        xa = bilinear_sampler(xa, grids)
        return xa, grids


class Encoder(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self.nce_layers = config['nce_layers']
        self.vgg = self.build_vgg()

    def call(self, x):
        return self.vgg(x)

    def build_vgg(self):
        vgg = VGG16(include_top=False)
        vgg.trainable = False
        outputs = [vgg.layers[idx].output for idx in self.nce_layers]
        return tf.keras.Model(inputs=vgg.input, outputs=outputs)


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


def affine_grid_generator(height, width, num_batch):
    theta = tf.concat([tf.repeat(tf.eye(2)[None, ...], num_batch, axis=0),
                       tf.zeros((num_batch, 2, 1))], axis=-1)
    # create normalized 2D grid
    x = tf.linspace(-1.0, 1.0, width)
    y = tf.linspace(-1.0, 1.0, height)
    x_t, y_t = tf.meshgrid(x, y)
    # flatten
    x_t_flat = tf.reshape(x_t, [-1])
    y_t_flat = tf.reshape(y_t, [-1])
    # reshape to [x_t, y_t , 1] - (homogeneous form)
    ones = tf.ones_like(x_t_flat)
    sampling_grid = tf.stack([x_t_flat, y_t_flat, ones])
    # repeat grid num_batch times
    sampling_grid = tf.expand_dims(sampling_grid, axis=0)
    sampling_grid = tf.tile(sampling_grid, tf.stack([num_batch, 1, 1]))

    # cast to float32 (required for matmul)
    theta = tf.cast(theta, 'float32')
    sampling_grid = tf.cast(sampling_grid, 'float32')

    # transform the sampling grid - batch multiply
    batch_grids = tf.matmul(theta, sampling_grid)
    # batch grid has shape (num_batch, 2, H*W)
    # reshape to (num_batch, H, W, 2)
    batch_grids = tf.reshape(batch_grids, [num_batch, 2, height, width])
    return batch_grids


def get_pixel_value(img, x, y):
    shape = tf.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]
    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
    b = tf.tile(batch_idx, (1, height, width))
    indices = tf.stack([b, y, x], 3)
    return tf.gather_nd(img, indices)


def bilinear_sampler(img, grids):
    x, y = grids[:, 0, ...], grids[:, 1, ...]
    H = tf.shape(img)[1]
    W = tf.shape(img)[2]
    max_y = tf.cast(H - 1, 'int32')
    max_x = tf.cast(W - 1, 'int32')
    zero = tf.zeros([], dtype='int32')

    # rescale x and y to [0, W-1/H-1]
    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')
    x = 0.5 * ((x + 1.0) * tf.cast(max_x - 1, 'float32'))
    y = 0.5 * ((y + 1.0) * tf.cast(max_y - 1, 'float32'))

    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    # clip to range [0, H-1/W-1] to not violate img boundaries
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    # get pixel value at corner coords
    Ia = get_pixel_value(img, x0, y0)
    Ib = get_pixel_value(img, x0, y1)
    Ic = get_pixel_value(img, x1, y0)
    Id = get_pixel_value(img, x1, y1)

    # recast as float for delta calculation
    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')

    # calculate deltas
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    # add dimension for addition
    wa = tf.expand_dims(wa, axis=3)
    wb = tf.expand_dims(wb, axis=3)
    wc = tf.expand_dims(wc, axis=3)
    wd = tf.expand_dims(wd, axis=3)

    # compute output
    out = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])

    return out


class InfoMatch(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self.CP = CoordPredictor(config)
        self.E = Encoder(config)
        self.F = PatchSampler(config) if config['loss_type']=='infonce' else None
        self.config=config

    def compile(self,
                G_optimizer,
                F_optimizer,
                D_optimizer):
        super().compile()
        self.G_optimizer = G_optimizer
        self.F_optimizer = F_optimizer
        self.D_optimizer = D_optimizer

        if self.config['loss_type']=='infonce':
            self.loss_func = PatchNCELoss(self.config['tau'])
        elif self.config['loss_type'] == 'perceptual_distance':
            self.loss_func = perceptual_loss
        elif self.config['loss_type'] == 'pixel_distance':
            self.loss_func = l1_loss

    @tf.function
    def train_step(self, inputs):
        xa, xb = inputs

        with tf.GradientTape(persistent=True) as tape:
            #translation
            xa_wrapped, _ = self.CP([xa, xb])
            #identity
            if self.config['use_identity']:
                xb_idt_wrapped, _ = self.CP([xb, xb])

            #comppute loss
            if self.config['loss_type'] == 'infonce':
                l_info_trl = self.loss_func(xb, xa_wrapped, self.E, self.F)
                l_info_idt = self.loss_func(xb, xb_idt_wrapped, self.E, self.F)\
                    if self.config['use_identity'] else 0.
            elif self.config['loss_type'] == 'perceptual_distance':
                l_info_trl = self.loss_func(xb, xa_wrapped, self.E)
                l_info_idt = self.loss_func(xb, xb_idt_wrapped, self.E)\
                    if self.config['use_identity'] else 0.

            elif self.config['loss_type'] == 'pixel_distance':
                l_info_trl = self.loss_func(xb, xa_wrapped)
                l_info_idt = self.loss_func(xb, xb_idt_wrapped) \
                    if self.config['use_identity'] else 0.

            #total loss
            loss = 0.5 * (l_info_trl + l_info_idt) \
                    if self.config['use_identity'] else l_info_trl

        grads = tape.gradient(loss, self.CP.trainable_weights +
                              self.F.trainable_weights if self.config['loss_type']=='infonce' else [])
        self.optimizer.apply_gradients(zip(grads, self.CP.trainable_weights +
                              self.F.trainable_weights if self.config['loss_type']=='infonce' else []))
        return {'info_trl': l_info_trl, 'info_idt':l_info_idt}

    @tf.function
    def test_step(self, inputs):
        xa, xb = inputs
        # translation
        xa_wrapped, _ = self.CP([xa, xb])
        # identity
        if self.config['use_identity']:
            xb_idt_wrapped, _ = self.CP([xb, xb])

        # comppute loss
        if self.config['loss_type'] == 'infonce':
            l_info_trl = self.loss_func(xb, xa_wrapped, self.E, self.F)
            l_info_idt = self.loss_func(xb, xb_idt_wrapped, self.E, self.F) \
                if self.config['use_identity'] else 0.
        elif self.config['loss_type'] == 'perceptual_distance':
            l_info_trl = self.loss_func(xb, xa_wrapped, self.E)
            l_info_idt = self.loss_func(xb, xb_idt_wrapped, self.E) \
                if self.config['use_identity'] else 0.

        elif self.config['loss_type'] == 'pixel_distance':
            l_info_trl = self.loss_func(xb, xa_wrapped)
            l_info_idt = self.loss_func(xb, xb_idt_wrapped) \
                if self.config['use_identity'] else 0.

        return {'info_trl': l_info_trl, 'info_idt':l_info_idt}
