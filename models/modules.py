from tensorflow.keras import layers
import tensorflow as tf

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
        if activation != 'none':
            self.activation = layers.Activation(activation)
        else:
            self.activation = tf.identity
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
        self.filters = filters
    def build(self, shape):
        dim = shape[-1]
        if dim != self.filters:
            self.skip = ConvBlock(self.filters,
                                  1,
                                  padding='valid',
                                  use_bias=False,
                                  norm_layer='none')
        else:
            self.skip = tf.identity

    def call(self, inputs, training=None):
        x = self.reflect_pad1(inputs)
        x = self.conv_block1(x)
        x = self.reflect_pad2(x)
        x = self.conv_block2(x)
        return x + self.skip(inputs)
