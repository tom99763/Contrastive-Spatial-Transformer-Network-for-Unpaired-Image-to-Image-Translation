import tensorflow as tf
from tensorflow.keras import losses

bc = losses.BinaryCrossentropy(from_logits=True)


def l_kl(h):
    return tf.reduce_mean(h ** 2)

def l1_loss(x, y):
    return tf.reduce_mean(tf.abs(x - y))


def l2_loss(x, y):
    return tf.reduce_mean((x - y) ** 2)


def perceptual_loss(source, target, netE):
    feat_source = netE(source, training=True)
    feat_target = netE(target, training=True)
    total_per_loss = 0.
    for feat_s, feat_t in zip(feat_source, feat_target):
        total_per_loss += l1_loss(feat_s, feat_t)
    return total_per_loss


def gan_loss(critic_real, critic_fake, gan_mode):
    if gan_mode == 'lsgan':
        d_loss = tf.reduce_mean((1 - critic_real) ** 2 + critic_fake ** 2)
        g_loss = tf.reduce_mean((1 - critic_fake) ** 2)

    elif gan_mode == 'logistic':
        d_loss = bc(tf.ones_like(critic_real),critic_real) + bc(tf.zeros_like(critic_fake),critic_fake)
        g_loss = bc(tf.ones_like(critic_fake),critic_fake)

    elif gan_mode == 'nonsaturate':
        d_loss = tf.reduce_mean(tf.math.softplus(-critic_real) + tf.math.softplus(critic_fake))
        g_loss = tf.reduce_mean(tf.math.softplus(-critic_fake))

    elif gan_mode == 'wgangp':
        d_loss = tf.reduce_mean(-critic_real + critic_fake)
        g_loss = tf.reduce_mean(-critic_fake)
    return 0.5 * d_loss, g_loss


class PatchNCELoss:
    def __init__(self, tau):
        # Potential: only supports for batch_size=1 now.
        self.tau = tau
        self.cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE,
            from_logits=True)

    def __call__(self, source, target, netE, netF):
        feat_source = netE(source, training=True)
        feat_target = netE(target, training=True)

        feat_source_pool, sample_ids = netF(feat_source, patch_ids=None, training=True)
        feat_target_pool, _ = netF(feat_target, patch_ids=sample_ids, training=True)

        total_nce_loss = 0.0
        for feat_s, feat_t in zip(feat_source_pool, feat_target_pool):
            n_patches, dim = feat_s.shape

            logit = tf.matmul(feat_s, tf.transpose(feat_t)) / self.tau
            # Diagonal entries are pos logits, the others are neg logits.
            diagonal = tf.eye(n_patches, dtype=tf.bool)
            target = tf.where(diagonal, 1.0, 0.0)

            loss = self.cross_entropy_loss(target, logit)
            total_nce_loss += tf.reduce_mean(loss)

        return total_nce_loss / len(feat_source_pool)



class PatchNCELoss_Dual:
    def __init__(self, tau):
        self.tau = tau
        self.cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE,
            from_logits=True)

    def __call__(self, source, target, netEx, netEy, netF, domain=['x', 'y']):
        feat_source = netEx(source, training=True)
        feat_target = netEy(target, training=True)

        feat_source_pool, sample_ids = netF(feat_source, patch_ids=None, training=True, domain=domain[0])
        feat_target_pool, _ = netF(feat_target, patch_ids=sample_ids, training=True, domain=domain[1])

        total_nce_loss = 0.0
        for feat_s, feat_t in zip(feat_source_pool, feat_target_pool):
            n_patches, dim = feat_s.shape

            logit = tf.matmul(feat_s, tf.transpose(feat_t)) / self.tau
            # Diagonal entries are pos logits, the others are neg logits.
            diagonal = tf.eye(n_patches, dtype=tf.bool)
            target = tf.where(diagonal, 1.0, 0.0)

            loss = self.cross_entropy_loss(target, logit)
            total_nce_loss += tf.reduce_mean(loss)

        return total_nce_loss / len(feat_source_pool)
