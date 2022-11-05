import argparse
from utils import *
from tensorflow.keras import optimizers

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', type=str, default='')
    parser.add_argument('--target_dir', type=str, default='')
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoints')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--model', type=str, default='CUT')
    parser.add_argument('--num_epochs', type=int, default=400)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--val_size', type=int, default=0.2)
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--beta_1', type=float, default=0.5, help='momentum of adam')
    parser.add_argument('--beta_2', type=float, default=0.99, help='momentum of adam')
    parser.add_argument('--num_samples', type=int, default=3)
    opt, _ = parser.parse_known_args()
    return opt
  
  
def main():
  opt = parse_opt()
  model, params = load_model(opt)
  
  ds_train, ds_val = build_dataset(opt)
  
  model.compile(G_optimizer = optimizers.Adam(learning_rate=opt.lr, beta_1=opt.beta_1, beta_2=opt.beta_2),
                F_optimizer = optimizers.Adam(learning_rate=opt.lr, beta_1=opt.beta_1, beta_2=opt.beta_2),
                D_optimizer = optimizers.Adam(learning_rate=opt.lr, beta_1=opt.beta_1, beta_2=opt.beta_2))
  
  ckpt_dir = f"{opt.ckpt_dir}/{opt.model}/{params}"
  if os.path.exists(ckpt_dir):
    model.load_weights(tf.train.latest_checkpoint(ckpt_dir))
    
  source, target = next(iter(ds_val.take(1)))
  callbacks = set_callbacks(opt, params, source[:opt.num_samples], target[:opt.num_samples])
  
  model.fit(
      x=ds_train,
      validation_data=ds_val,
      epochs=opt.num_epochs,
      callbacks=callbacks
  )
  
if __name__ == '__main__':
    main()
    
    
