import tensorflow as tf
from tensorflow.keras import callbacks
import matplotlib.pyplot as plt

def learned_patch_relation(x, E, target_id):
  feature_maps = E(x)
  fig, ax = plt.subplots(ncols = x.shape[0], nrows = 1 + len(feature_maps))
    
  
def quantitive_visualize(x, model, opt, params):
  #y.shape[0]=15
  y = model.G(x)
  fig, ax = plt.subplots(ncols = 5, nrows = 6, figsize=(12, 12))
  
  k=0
  for i in range(3):
    for j in range(5):
      ax[2*i, j].imshow(x[k] * 0.5 + 0.5)
      ax[2*i, j].axis('off')
      ax[2*i+1, j].imshow(y[k] * 0.5 + 0.5)
      ax[2*i+1, j].axis('off')
      k+=1
  plt.savefig(f'{opt.output_dir}/{opt.model}/params.png')
