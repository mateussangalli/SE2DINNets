import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, regularizers
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import os
import argparse


from load_data import load_data


mnist_rot_dir = 'mnist_rot'
mnist_12k_dir = 'mnist12k'
model_dir = 'models'




parser = argparse.ArgumentParser(description='trains a SE2DINNet with the specified parameters(some
                                                parts of the architecture are fixed)')
parser.add_argument('-o', '--order', type=int, default=2, help='order of the differential invariants')
parser.add_argument('-d', '--dropout', type=int, default=30, help='dropout rate in percentage
                                                                    between 1 x 1 convolutions')
parser.add_argument('-w', '--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('-f', '--data_dir', type=str, default='./', help='directory containing both
                                          MNIST-Rot and MNIST12K dataset in separate folders')
parser.add_argument('--train_on_mnist12k', action='store_true', help='whether to train on MNIST12K
                                                                    or MNIST-Rot(False)')
parser.add_argument('--lr', type=float, default=1e-2, help='initial learning rate')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--epochs', type=int, default=2000, help='maximum number of epochs')
parser.add_argument('--n_filters', type=int, default=20, help='number of filters in the middle layers')

args = parser.parse_args()

weight_decay = args.weight_decay
dropout = args.dropout / 100
n_filters = args.n_filters
lr = args.lr
batch_size = args.batch_size
epochs = args.epochs
order = args.order
data_dir = args.data_dir

if args.train_on_mnist12k:
  _, _, (x_test, y_test) = load_data(os.path.join(data_dir, mnist_rot_dir)
  (x_train, y_train), (x_val, y_val), _ = load_data(os.path.join(data_dir, mnist_12k_dir))
else:
  (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data(os.path.join(data_dir, mnist_rot_dir))

def se2din_block(n_in, n_out, sigma, width, order=2, dropout=0, weight_decay=0):
  block = tf.keras.models.Sequential()
  block.add(layers.Input((None, None, n_in)))
  block.add(DISE2(sigma, width, order=order))
  block.add(layers.BatchNormalization(beta_regularizer=regularizers.l2(weight_decay),
                gamma_regularizer=regularizers.l2(weight_decay)))

  block.add(layers.Conv2D(n_out,1,
                  kernel_regularizer=regularizers.l2(weight_decay),
                  bias_regularizer=regularizers.l2(weight_decay)))
  block.add(layers.BatchNormalization(beta_regularizer=regularizers.l2(weight_decay),
                gamma_regularizer=regularizers.l2(weight_decay)))
  block.add(layers.ReLU())
  if dropout > 0:
    block.add(layers.Dropout(dropout))

  block.add(layers.Conv2D(n_out,1,
                  kernel_regularizer=regularizers.l2(weight_decay),
                  bias_regularizer=regularizers.l2(weight_decay)))
  block.add(layers.BatchNormalization(beta_regularizer=regularizers.l2(weight_decay),
                gamma_regularizer=regularizers.l2(weight_decay)))
                
  #block.add(layers.ReLU())
  return block

def get_model(n_filters, weight_decay, dropout, lr, order=2):
  input_layer = layers.Input((None,None,1))
  
  
  x = se2din_block(1,n_filters,1.,4,2,dropout,weight_decay)(input_layer)
  features0 = tf.keras.models.Model(input_layer, x)
  x += se2din_block(n_filters,n_filters,1.,4,order,dropout,weight_decay)(x)
  x += se2din_block(n_filters,n_filters,2.,8,order,dropout,weight_decay)(x)
  x += se2din_block(n_filters,n_filters,2.,8,order,dropout,weight_decay)(x)
  x += se2din_block(n_filters,n_filters,2.,8,order,dropout,weight_decay)(x)
  
  x = se2din_block(n_filters,10,2.,8,2,0,bn_momentum,weight_decay)(x)
  
  features1 = tf.keras.models.Model(input_layer, x)
  
  x = layers.GlobalMaxPooling2D()(x)
  
  
  model = tf.keras.models.Model(input_layer, x)
  model.summary()
  model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.Adam(lr), metrics=['accuracy'])
  return model

  


model = get_model(n_filters, weight_decay, dropout, lr, order)

# reduces learning rate when validation loss stagnates
cb_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.1, patience=100, verbose=0,
    mode='auto', min_delta=0.0001, min_lr=1e-5
    )

# stops training if validation loss remains unchanged for too long
cb_es = tf.keras.callbacks.EarlyStopping(
   monitor='val_loss', min_delta=0, patience=300, verbose=0,
   mode='auto', restore_best_weights=True
   )
  
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=[cb_lr, cb_es], verbose=2)
model.evaluate(x_test, y_test)


model.save(os.path.join(model_dir, f'SE2DINNetOrd{order}'))
