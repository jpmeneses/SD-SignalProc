import os
import numpy as np
import scipy.io as sio
import tensorflow as tf
import tensorflow.keras as keras
import tqdm

import DLlib as dl
import pylib as py
import tf2lib as tl

py.arg('--experiment_dir', default='IMU-Seg')
py.arg('--n_G_filters', type=int, default=36)
py.arg('--n_downsamplings', type=int, default=4)
py.arg('--n_res_blocks', type=int, default=2)
py.arg('--encoded_size', type=int, default=256)
py.arg('--batch_size', type=int, default=64)
py.arg('--epochs', type=int, default=100)
py.arg('--epoch_decay', type=int, default=100)  # epoch to start decaying learning rate
py.arg('--epoch_ckpt', type=int, default=20)  # num. of epochs to save a checkpoint
py.arg('--lr', type=float, default=0.001)
py.arg('--main_loss', default='MSE', choices=['MSE', 'MAE', 'MSLE'])
py.arg('--A_loss_weight', type=float, default=1.0)
py.arg('--cov_reg_weight', type=float, default=0.0)
py.arg('--pool_size', type=int, default=50)  # pool size to store fake samples
args = py.args()

# output_dir
output_dir = py.join('output',args.experiment_dir)
py.mkdir(output_dir)

# save settings
py.args_to_yaml(py.join(output_dir, 'settings.yml'), args)

file_dir = '../Data/'
pat_filenames = ['01','03']#,'14']

X,y = tl.load_data(file_dir, pat_filenames)
len_dataset,wdt,n_f,n_ch = np.shape(X)

print('Training input shape:', X.shape)
print('Training output shape:', len(y))
print('Max. X value', np.max(X))
print('Min. X value', np.min(X))
print('X dtype:',X.dtype)

A_dataset = tf.data.Dataset.from_tensor_slices((X))
A_dataset = A_dataset.batch(args.batch_size).shuffle(len_dataset)

enc= dl.encoder(input_shape=(wdt,n_f,n_ch),
                encoded_dims=args.encoded_size,
                filters=args.n_G_filters,
                num_layers=args.n_downsamplings,
                num_res_blocks=args.n_res_blocks,
                )
dec= dl.decoder(encoded_dims=args.encoded_size,
                output_shape=(wdt,n_f,n_ch),
                filters=args.n_G_filters,
                num_layers=args.n_downsamplings,
                num_res_blocks=args.n_res_blocks,
                )

Cov_op = dl.CoVar()

if args.main_loss == 'MSE':
  cycle_loss_fn = tf.losses.MeanSquaredError()
elif args.main_loss == 'MAE':
  cycle_loss_fn = tf.losses.MeanAbsoluteError()
elif args.main_loss == 'MSLE':
  cycle_loss_fn = tf.losses.MeanSquaredLogarithmicError()
else:
  raise(NameError('Unrecognized Main Loss Function'))

G_optimizer = keras.optimizers.Adam(learning_rate=args.lr)

@tf.function
def train_G(A):
  with tf.GradientTape() as t:
    ##################### A Cycle #####################
    A2Z = enc(A, training=True)
    A2Z2A = dec(A2Z, training=False)
    A2Z2A_cycle_loss = cycle_loss_fn(A, A2Z2A)

    A2Z_cov = Cov_op(A2Z, training=False)
    A2Z_cov_loss = cycle_loss_fn(A2Z_cov,tf.eye(A2Z_cov.shape[0]))

    G_loss = args.A_loss_weight * A2Z2A_cycle_loss + args.cov_reg_weight * A2Z_cov_loss

  G_grad = t.gradient(G_loss, enc.trainable_variables + dec.trainable_variables)
  G_optimizer.apply_gradients(zip(G_grad, enc.trainable_variables + dec.trainable_variables))
  
  return A2Z2A,  {'A2Z2A_cycle_loss': A2Z2A_cycle_loss,
                  'Cov_reg': A2Z_cov_loss,}


def train_step(A):
  A2Z2A, G_loss_dict = train_G(A)
  return G_loss_dict


# epoch counter
ep_cnt = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)

# checkpoint
checkpoint = tl.Checkpoint(dict(enc=enc,
                                dec=dec,
                                G_optimizer=G_optimizer,
                                ep_cnt=ep_cnt),
                           py.join(output_dir, 'checkpoints'),
                           max_to_keep=5)
try:  # restore checkpoint including the epoch counter
    checkpoint.restore().assert_existing_objects_matched()
except Exception as e:
    print(e)

# summary
train_summary_writer = tf.summary.create_file_writer(py.join(output_dir, 'summaries', 'train'))

# main loop
for ep in range(args.epochs):
  if ep < ep_cnt:
    continue

  # update epoch counter
  ep_cnt.assign_add(1)

  # train for an epoch
  for A in tqdm.tqdm(A_dataset, desc='Ep-%03d' % (ep+1), total=len_dataset//args.batch_size):
    G_loss_dict = train_step(A)

  # summary
  with train_summary_writer.as_default():
    tl.summary(G_loss_dict, step=G_optimizer.iterations, name='G_losses')

  # save checkpoint
  if (((ep+1) % args.epoch_ckpt) == 0) or ((ep+1)==args.epochs):
    checkpoint.save(ep)
