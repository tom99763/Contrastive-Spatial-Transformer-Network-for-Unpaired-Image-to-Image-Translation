import os
import tensorflow as tf
from sklearn.model_selection import train_test_split as ttp
from models import CUT, InfoMatch, CycleGAN, UNIT, UGATIT
from tensorflow.keras import callbacks
import matplotlib.pyplot as plt
import yaml
from metrics import metrics
import numpy as np

AUTOTUNE = tf.data.experimental.AUTOTUNE

def load_model(opt):
    config = get_config(f'./configs/{opt.model}.yaml')
    if opt.model == 'CUT':
        model = CUT.CUT(config, opt)
        params = f"{config['tau']}_{config['lambda_nce']}_{config['use_identity']}"

    elif opt.model == 'InfoMatch':
        model = InfoMatch.InfoMatch(config, opt)
        params = f"{config['loss_type']}_{config['tau']}_{config['use_identity']}"

    elif opt.model == 'CycleGAN':
        model = CycleGAN.CycleGAN(config, opt)
        params='_'

    elif opt.model == 'UNIT':
        model = UNIT.UNIT(config, opt)
        params='_'

    elif opt.model == 'UGATIT':
        model = UGATIT.UGATIT(config, opt)
        params='_'

    return model, params


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)

def get_image(pth, opt):
    image = tf.image.decode_jpeg(tf.io.read_file(pth), channels=opt.num_channels)
    image = tf.cast(tf.image.resize(image, (opt.image_size, opt.image_size)), 'float32')
    return (image-127.5)/127.5

def build_tf_dataset(source_list, target_list, opt):
    ds_source = tf.data.Dataset.from_tensor_slices(source_list).map(lambda pth: get_image(pth, opt),
                                                                    num_parallel_calls=AUTOTUNE).shuffle(256).prefetch(
        AUTOTUNE)
    ds_target = tf.data.Dataset.from_tensor_slices(target_list).map(lambda pth: get_image(pth, opt),
                                                                    num_parallel_calls=AUTOTUNE).shuffle(256).prefetch(
        AUTOTUNE)
    ds = tf.data.Dataset.zip((ds_source, ds_target)).shuffle(256).batch(opt.batch_size, drop_remainder=True).prefetch(
        AUTOTUNE)
    return ds


def build_dataset(opt):
    source_list = list(map(lambda x: f'{opt.source_dir}/{x}', os.listdir(opt.source_dir)))
    target_list = list(map(lambda x: f'{opt.target_dir}/{x}', os.listdir(opt.target_dir)))
    length = min(len(source_list), len(target_list))
    source_list = source_list[:length]
    target_list = target_list[:length]

    source_train, source_val, target_train, target_val = ttp(source_list, target_list, test_size=opt.val_size,
                                                             random_state=999, shuffle=True)

    ds_train = build_tf_dataset(source_train, target_train, opt)
    ds_val = build_tf_dataset(source_val, target_val, opt)

    return ds_train, ds_val

def makecolorwheel():
    # Create a colorwheel for visualization
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros((ncols, 3))

    col = 0
    # RY
    colorwheel[0:RY, 0] = 1
    colorwheel[0:RY, 1] = np.arange(0, 1, 1. / RY)
    col += RY

    # YG
    colorwheel[col:col + YG, 0] = np.arange(1, 0, -1. / YG)
    colorwheel[col:col + YG, 1] = 1
    col += YG

    # GC
    colorwheel[col:col + GC, 1] = 1
    colorwheel[col:col + GC, 2] = np.arange(0, 1, 1. / GC)
    col += GC

    # CB
    colorwheel[col:col + CB, 1] = np.arange(1, 0, -1. / CB)
    colorwheel[col:col + CB, 2] = 1
    col += CB

    # BM
    colorwheel[col:col + BM, 2] = 1
    colorwheel[col:col + BM, 0] = np.arange(0, 1, 1. / BM)
    col += BM

    # MR
    colorwheel[col:col + MR, 2] = np.arange(1, 0, -1. / MR)
    colorwheel[col:col + MR, 0] = 1

    return colorwheel

def viz_flow(u, v, logscale=True, scaledown=6, output=False):
    colorwheel = makecolorwheel()
    ncols = colorwheel.shape[0]
    radius = np.sqrt(u ** 2 + v ** 2)
    radius = radius / scaledown
    rot = np.arctan2(-v, -u) / np.pi
    fk = (rot + 1) / 2 * (ncols - 1)  # -1~1 maped to 0~ncols
    k0 = fk.astype(np.uint8)  # 0, 1, 2, ..., ncols
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    ncolors = colorwheel.shape[1]
    img = np.zeros(u.shape + (ncolors,))
    for i in range(ncolors):
        tmp = colorwheel[:, i]
        col0 = tmp[k0]
        col1 = tmp[k1]
        col = (1 - f) * col0 + f * col1

        idx = radius <= 1
        # increase saturation with radius
        col[idx] = 1 - radius[idx] * (1 - col[idx])
        # out of range
        col[~idx] *= 0.75
        img[:, :, i] = np.floor(255 * col)
    return img/255.

###Callbacks
class VisualizeCallback(callbacks.Callback):
    def __init__(self, source, target, opt, params):
        super().__init__()
        self.source = source
        self.target = target
        self.opt = opt
        self.params_ = params

    def on_epoch_end(self, epoch, logs=None):
        b, h, w, c = self.target.shape

        if self.opt.model == 'InfoMatch':
            x2y_wrapped, grids = self.model.CP(self.source)
            x2y, rxy = self.model.R(x2y_wrapped)
            grids = tf.transpose(grids, [0, 2, 3, 1])

        elif self.opt.model == 'CycleGAN':
            x2y = self.model.Gb(self.source)

        elif self.opt.model == 'UNIT':
            ha, _ = self.model.Ga.encode(self.source)
            x2y = self.model.Gb.decode(ha)

        elif self.opt.model == 'UGATIT':
            x2y, _ = self.model.Gb(self.source)

        else:
            x2y = self.model.G(self.source)

        fig, ax = plt.subplots(ncols=b, nrows=5 if self.opt.model == 'InfoMatch' else 2, figsize=(16, 16))

        for i in range(b):

            if self.opt.model == 'InfoMatch':
                ax[0, i].imshow(self.source[i] * 0.5 + 0.5, cmap='gray')
                ax[0, i].axis('off')
                ax[1, i].imshow(x2y_wrapped[i] * 0.5 + 0.5, cmap='gray')
                ax[1, i].axis('off')
                ax[2, i].imshow(rxy[i] * 0.5 + 0.5, cmap='gray')
                ax[2, i].axis('off')
                ax[3, i].imshow(x2y[i] * 0.5 + 0.5, cmap='gray')
                ax[3, i].axis('off')
                grid_img = viz_flow(grids[i, ..., 0], grids[i, ..., 1])
                ax[4, i].imshow(grid_img)
                ax[4, i].axis('off')
            else:
                ax[0, i].imshow(self.source[i] * 0.5 + 0.5)
                ax[0, i].axis('off')
                ax[1, i].imshow(x2y[i] * 0.5 + 0.5)
                ax[1, i].axis('off')
        plt.tight_layout()
        dir = f'{self.opt.output_dir}/{self.opt.model}/{self.params_}'
        if not os.path.exists(dir):
            os.makedirs(dir)
        plt.savefig(f'{dir}/synthesis_{epoch}.jpg')


def set_callbacks(opt, params, source, target, val_ds = None):
    ckpt_dir = f"{opt.ckpt_dir}/{opt.model}"
    output_dir = f"{opt.output_dir}/{opt.model}"

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    checkpoint_callback = callbacks.ModelCheckpoint(filepath=f"{ckpt_dir}/{params}/{opt.model}", save_weights_only=True)
    history_callback = callbacks.CSVLogger(f"{output_dir}/{params}.csv", separator=",", append=False)
    visualize_callback = VisualizeCallback(source, target, opt, params)
    metrics_callback = metrics.MetricsCallbacks(val_ds, opt, params)
    return [checkpoint_callback, history_callback, visualize_callback, metrics_callback]
