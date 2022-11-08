from tensorflow.keras import layers
import tensorflow as tf
import numpy as np

class Padding2D(layers.Layer):
    """ 2D padding layer.
    """
    def __init__(self, padding=(1, 1), pad_type='constant', **kwargs):
        assert pad_type in ['constant', 'reflect', 'symmetric']
        super(Padding2D, self).__init__(**kwargs)
        self.padding = (padding, padding) if type(padding) is int else tuple(padding)
        self.pad_type = pad_type

    def call(self, inputs, training=None):
        padding_width, padding_height = self.padding
        padding_tensor = [
            [0, 0],
            [padding_height, padding_height],
            [padding_width, padding_width],
            [0, 0],
        ]

        return tf.pad(inputs, padding_tensor, mode=self.pad_type)


class InstanceNorm(layers.Layer):
    def __init__(self, epsilon=1e-5, affine=False, **kwargs):
        super(InstanceNorm, self).__init__(**kwargs)
        self.epsilon = epsilon
        self.affine = affine
 
    def build(self, input_shape):
        if self.affine:
            self.gamma = self.add_weight(name='gamma',
                                        shape=(input_shape[-1],),
                                        initializer=tf.random_normal_initializer(0, 0.02),
                                        trainable=True)
            self.beta = self.add_weight(name='beta',
                                        shape=(input_shape[-1],),
                                        initializer=tf.zeros_initializer(),
                                        trainable=True)

    def call(self, inputs, training=None):
        mean, var = tf.nn.moments(inputs, axes=[1, 2], keepdims=True)
        x = tf.divide(tf.subtract(inputs, mean), tf.math.sqrt(tf.add(var, self.epsilon)))
        if self.affine:
            return self.gamma * x + self.beta
        return x
 

class ConvBlock(layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1,1),
                 padding='valid',
                 use_bias=True,
                 norm_layer=None,
                 activation='linear',
                 **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        initializer = tf.random_normal_initializer(0., 0.02)
        self.conv2d = layers.Conv2D(filters,
                             kernel_size,
                             strides,
                             padding,
                             use_bias=use_bias,
                             kernel_initializer=initializer)
        self.activation = layers.Activation(activation)
        if norm_layer == 'batch':
            self.normalization = layers.BatchNormalization()
        elif norm_layer == 'instance':
            self.normalization = InstanceNorm(affine=False)
        else:
            self.normalization = tf.identity

    def call(self, inputs, training=None):
        x = self.conv2d(inputs)
        x = self.normalization(x)
        x = self.activation(x)
        return x
      
class ConvTransposeBlock(layers.Layer):
    """ ConvTransposeBlock layer consists of Conv2DTranspose + Normalization + Activation.
    """
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1,1),
                 padding='valid',
                 use_bias=True,
                 norm_layer=None,
                 activation='linear',
                 **kwargs):
        super(ConvTransposeBlock, self).__init__(**kwargs)
        initializer = tf.random_normal_initializer(0., 0.02)
        self.convT2d = layers.Conv2DTranspose(filters,
                                       kernel_size,
                                       strides,
                                       padding,
                                       use_bias=use_bias,
                                       kernel_initializer=initializer)
        self.activation = layers.Activation(activation)
        if norm_layer == 'batch':
            self.normalization = layers.BatchNormalization()
        elif norm_layer == 'instance':
            self.normalization = InstanceNorm(affine=False)
        else:
            self.normalization = tf.identity

    def call(self, inputs, training=None):
        x = self.convT2d(inputs)
        x = self.normalization(x)
        x = self.activation(x)
        return x


class ResBlock(layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 use_bias,
                 norm_layer,
                 **kwargs):
        super(ResBlock, self).__init__(**kwargs)

        self.reflect_pad1 = Padding2D(1, pad_type='reflect')
        self.conv_block1 = ConvBlock(filters,
                                     kernel_size,
                                     padding='valid',
                                     use_bias=use_bias,
                                     norm_layer=norm_layer,
                                     activation='relu')

        self.reflect_pad2 = Padding2D(1, pad_type='reflect')
        self.conv_block2 = ConvBlock(filters,
                                     kernel_size,
                                     padding='valid',
                                     use_bias=use_bias,
                                     norm_layer=norm_layer)

    def call(self, inputs, training=None):
        x = self.reflect_pad1(inputs)
        x = self.conv_block1(x)
        x = self.reflect_pad2(x)
        x = self.conv_block2(x)
        return inputs + x
    
    
### feature extraction
def get_sobel_kernel(ksize):
    if (ksize % 2 == 0) or (ksize < 1):
        raise ValueError("Kernel size must be a positive odd number")
    _base = np.arange(ksize) - ksize//2
    a = np.broadcast_to(_base, (ksize,ksize))
    b = ksize//2 - np.abs(a).T
    s = np.sign(a)
    return a + s*b


def get_gaussian_kernel(ksize = 3, sigma = -1.0):
    ksigma = 0.15*ksize + 0.35 if sigma <= 0 else sigma
    i, j   = np.mgrid[0:ksize,0:ksize] - (ksize-1)//2
    kernel = np.exp(-(i**2 + j**2) / (2*ksigma**2))
    return kernel / kernel.sum()


def get_laplacian_of_gaussian_kernel(ksize = 3, sigma = -1.0):
    ksigma = 0.15*ksize + 0.35 if sigma <= 0 else sigma
    i, j   = np.mgrid[0:ksize,0:ksize] - (ksize-1)//2
    kernel = (i**2 + j**2 - 2*ksigma**2) / (ksigma**4) * np.exp(-(i**2 + j**2) / (2*ksigma**2))
    return kernel - kernel.mean()


def tf_kernel_prep_4d(kernel, n_channels):
    return np.tile(kernel, (n_channels, 1, 1, 1)).swapaxes(0,2).swapaxes(1,3)


def tf_kernel_prep_3d(kernel, n_channels):
    return np.tile(kernel, (n_channels, 1, 1)).swapaxes(0,1).swapaxes(1,2)


def tf_filter2d(batch, kernel, strides=(1,1), padding='SAME'):
    n_ch = batch.shape[3].value
    tf_kernel = tf.constant(tf_kernel_prep_4d(kernel, n_ch))
    return tf.nn.depthwise_conv2d(batch, tf_kernel, [1, strides[0], strides[1], 1], padding=padding)

 
def tf_deriv(batch, ksize=3, padding='SAME'):
    try:
        n_ch = batch.shape[3].value
    except:
        n_ch = int(batch.get_shape()[3])
    gx = tf_kernel_prep_3d(np.array([[ 0, 0, 0],
                                     [-1, 0, 1],
                                     [ 0, 0, 0]]), n_ch)    
    gy = tf_kernel_prep_3d(np.array([[ 0, -1, 0],
                                     [ 0, 0, 0],
                                     [ 0, 1, 0]]), n_ch)   
    kernel = tf.constant(np.stack([gx, gy], axis=-1), name="DerivKernel", dtype = np.float32)
    return tf.nn.depthwise_conv2d(batch, kernel, [1, 1, 1, 1], padding=padding, name="GradXY")
    

def tf_sobel(batch, ksize=3, padding='SAME'):
    n_ch = batch.shape[3].value
    gx = tf_kernel_prep_3d(get_sobel_kernel(ksize),   n_ch)
    gy = tf_kernel_prep_3d(get_sobel_kernel(ksize).T, n_ch)
    kernel = tf.constant(np.stack([gx, gy], axis=-1), dtype = np.float32)
    return tf.nn.depthwise_conv2d(batch, kernel, [1, 1, 1, 1], padding=padding)


def tf_sharr(batch, ksize=3, padding='SAME'):
    n_ch = batch.shape[3].value
    gx = tf_kernel_prep_3d([[ -3, 0,  3],
                            [-10, 0, 10],
                            [ -3, 0,  3]], n_ch)    
    gy = tf_kernel_prep_3d([[-3,-10,-3],
                            [ 0,  0, 0],
                            [ 3, 10, 3]], n_ch)    
    kernel = tf.constant(np.stack([gx, gy], axis=-1), dtype = np.float32)
    return tf.nn.depthwise_conv2d(batch, kernel, [1, 1, 1, 1], padding=padding)


def tf_laplacian(batch, padding='SAME'):
    kernel = np.array([[0, 1, 0],
                       [1,-4, 1],
                       [0, 1, 0]], dtype=batch.dtype)    
    return tf_filter2d(batch, kernel, padding=padding)


def tf_boxfilter(batch, ksize = 3, padding='SAME'):
    kernel = np.ones((ksize, ksize), dtype=batch.dtype) / ksize**2
    return tf_filter2d(batch, kernel, padding=padding)

def tf_rad2deg(rad):
    return 180 * rad / tf.constant(np.pi)

def tf_select_by_idx(a, idx, grayscale):
    if grayscale:
        return a[:,:,:,0]
    else:
        return tf.where(tf.equal(idx, 2), 
                         a[:,:,:,2], 
                         tf.where(tf.equal(idx, 1), 
                                   a[:,:,:,1], 
                                   a[:,:,:,0]))
    

def tf_hog_descriptor(images, cell_size = 8, block_size = 2, block_stride = 1, n_bins = 9,
                      grayscale = False):

    batch_size, height, width, depth = images.shape
    scale_factor = tf.constant(180 / n_bins, name="scale_factor", dtype=tf.float32)
    
    img = tf.constant(images, name="ImgBatch", dtype=tf.float32)

    if grayscale:
        img = tf.image.rgb_to_grayscale(img, name="ImgGray")

    # automatically padding height and width to valid size (multiples of cell size)
    if height % cell_size != 0 or width % cell_size != 0:
        height = height + (cell_size - (height % cell_size)) % cell_size
        width = width + (cell_size - (width % cell_size)) % cell_size
        img = tf.image.resize_image_with_crop_or_pad(img, height, width)
    
    # gradients
    grad = tf_deriv(img)
    g_x = grad[:,:,:,0::2]
    g_y = grad[:,:,:,1::2]
    
    # masking unwanted gradients of edge pixels
    mask_depth = 1 if grayscale else depth
    g_x_mask = np.ones((batch_size, height, width, mask_depth))
    g_y_mask = np.ones((batch_size, height, width, mask_depth))
    g_x_mask[:, :, (0, -1)] = 0
    g_y_mask[:, (0, -1)] = 0
    g_x_mask = tf.constant(g_x_mask, dtype=tf.float32)
    g_y_mask = tf.constant(g_y_mask, dtype=tf.float32)
    
    g_x = g_x*g_x_mask
    g_y = g_y*g_y_mask

    # maximum norm gradient selection
    g_norm = tf.sqrt(tf.square(g_x) + tf.square(g_y), "GradNorm")
    
    if not grayscale and depth != 1:
        # maximum norm gradient selection
        idx    = tf.argmax(g_norm, 3)
        g_norm = tf.expand_dims(tf_select_by_idx(g_norm, idx, grayscale), -1)
        g_x    = tf.expand_dims(tf_select_by_idx(g_x,    idx, grayscale), -1)
        g_y    = tf.expand_dims(tf_select_by_idx(g_y,    idx, grayscale), -1)

    g_dir = tf_rad2deg(tf.atan2(g_y, g_x)) % 180
    g_bin = tf.to_int32(g_dir / scale_factor, name="Bins")

    # cells partitioning
    cell_norm = tf.space_to_depth(g_norm, cell_size, name="GradCells")
    cell_bins = tf.space_to_depth(g_bin,  cell_size, name="BinsCells")

    # cells histograms
    hist = list()
    zero = tf.zeros(cell_bins.get_shape()) 
    for i in range(n_bins):
        mask = tf.equal(cell_bins, tf.constant(i, name="%i"%i))
        hist.append(tf.reduce_mean(tf.where(mask, cell_norm, zero), 3))
    hist = tf.transpose(tf.stack(hist), [1,2,3,0], name="Hist")

    # blocks partitioning
    block_hist = tf.extract_image_patches(hist, 
                                          ksizes  = [1, block_size, block_size, 1], 
                                          strides = [1, block_stride, block_stride, 1], 
                                          rates   = [1, 1, 1, 1], 
                                          padding = 'VALID',
                                          name    = "BlockHist")

    # block normalization
    block_hist = tf.nn.l2_normalize(block_hist, 3, epsilon=1.0)
    
    # HOG descriptor
    hog_descriptor = tf.reshape(block_hist, 
                                [int(block_hist.get_shape()[0]), 
                                 int(block_hist.get_shape()[1]) * \
                                 int(block_hist.get_shape()[2]) * \
                                 int(block_hist.get_shape()[3])], 
                                 name='HOGDescriptor')

    return hog_descriptor, block_hist, hist


