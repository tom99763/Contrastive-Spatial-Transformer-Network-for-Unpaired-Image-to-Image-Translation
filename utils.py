import os
import tensorflow as tf
from sklearn.model_selection import train_test_split as ttp
from models import CUT
from tensorflow.keras import callbacks
import matplotlib.pyplot as plt
import yaml

AUTOTUNE = tf.data.experimental.AUTOTUNE

def load_model(opt):
    config = get_config(f'./configs/{opt.model}.yaml')
    if opt.model == 'CUT':
        model = CUT.CUT(config)
        params = f"{config['tau']}_{config['lambda_nce']}_{config['use_identity']}"
    return model, params
     

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)


def get_image(pth, opt):
    image = tf.image.decode_jpeg(tf.io.read_file(pth), channels=3)
    image = tf.cast(tf.image.resize(image, (opt.image_size, opt.image_size)), 'float32')
    return (image-127.5)/127.5


def build_tf_dataset(source_list, target_list, opt):
  ds_source = tf.data.Dataset.from_tensor_slices(source_list).map(lambda pth : get_image(pth, opt), num_parallel_calls=AUTOTUNE).shuffle(256).prefetch(AUTOTUNE)
  ds_target = tf.data.Dataset.from_tensor_slices(target_list).map(lambda pth : get_image(pth, opt), num_parallel_calls=AUTOTUNE).shuffle(256).prefetch(AUTOTUNE)
  ds = tf.data.Dataset.zip((ds_source, ds_target)).shuffle(256).batch(opt.batch_size, drop_remainder=True).prefetch(AUTOTUNE)
  return ds
  
def build_dataset(opt):
  source_list = list(map(lambda x: f'{opt.source_dir}/{x}' , os.listdir(opt.source_dir)))
  target_list = list(map(lambda x: f'{opt.target_dir}/{x}' , os.listdir(opt.target_dir)))
  
  source_train, source_val, target_train, target_val = ttp(source_list, target_list , test_size = opt.val_size, random_state =999, shuffle=True)
  
  ds_train = build_tf_dataset(source_train, target_train, opt)
  ds_val = build_tf_dataset(source_val, target_val, opt)
  
  return ds_train, ds_val


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
        xa_repeat = tf.repeat(self.source, b, axis=0)
        xb_repeat = tf.reshape(tf.stack([self.target for _ in range(b)], axis=0), (b ** 2, h, w, c))
        
        try:
            ca, sa = self.model.G.encode(xa_repeat)
            cb, sb = self.model.G.encode(xb_repeat)
            xa2b = self.model.G.decode(ca, sb)
        except:
            ca, sa = self.model.Ga.encode(xa_repeat)
            cb, sb = self.model.Gb.encode(xb_repeat)
            xa2b = self.model.Gb.decode(ca, sb)
        
        source = self.source * 0.5 + 0.5
        target = self.target * 0.5 + 0.5
        xa2b = xa2b * 0.5 + 0.5
        
        fig, ax = plt.subplots(ncols=b + 1, nrows=b + 1, figsize=(8, 8))
        for k in range(b + 1):
            if k == 0:
                ax[0, k].imshow(tf.ones(source[0].shape))
                ax[0, k].axis('off')
            else:
                ax[0, k].imshow(source[k - 1])
                ax[0, k].axis('off')
                
        for k in range(1, b + 1):
            ax[k, 0].imshow(target[k - 1])
            ax[k, 0].axis('off')
                
        k = 0
        for j in range(b):
            for i in range(b):
                ax[i + 1, j + 1].imshow(xa2b[k])
                ax[i + 1, j + 1].axis('off')
                k += 1
        plt.tight_layout()
        dir = f'{self.opt.output_dir}/{self.opt.model}/{self.params_}'
        if not os.path.exists(dir):
            os.makedirs(dir)
        plt.savefig(f'{dir}/synthesis_{epoch}.jpg')


def set_callbacks(opt, params, source, target):
    ckpt_dir = f"{opt.ckpt_dir}/{opt.model}"
    output_dir = f"{opt.output_dir}/{opt.model}"
    
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    checkpoint_callback = callbacks.ModelCheckpoint(filepath=f"{ckpt_dir}/{params}/{opt.model}", save_weights_only=True)
    history_callback = callbacks.CSVLogger(f"{output_dir}/{params}.csv",separator="," , append=False)
    visualize_callback = VisualizeCallback(source, target, opt, params)
    return [checkpoint_callback, history_callback, visualize_callback]
