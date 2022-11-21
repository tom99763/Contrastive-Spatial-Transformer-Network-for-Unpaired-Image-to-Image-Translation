import tensorflow as tf

def fft(x):
    x = tf.transpose(x, perm=[0, 3, 1, 2])
    fx = tf.signal.rfft(x)
    return fx


def cross_correlation(x1, x2):
    fx1 = fft(x1)
    fx2 = fft(x2)
    fr = fx1 * tf.math.conj(fx2)
    r = tf.signal.irfft(fr)
    r = tf.transpose(r, perm=[0, 2, 3, 1])
    return r
