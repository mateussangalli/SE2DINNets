import numpy as np
import os


def load_data(data_dir):
  """
  loads the training, validation and test sets for either the mnist-rot or mnist12k datasets
  """
  with np.load(os.path.join(data_dir, 'train.npz')) as data:
    x_train = data['data']
    y_train = data['labels'].astype(np.int32)

  with np.load(os.path.join(data_dir, 'valid.npz')) as data:
    x_val = data['data']
    y_val = data['labels'].astype(np.int32)

  with np.load(os.path.join(data_dir, 'test.npz')) as data:
    x_test = data['data']
    y_test = data['labels'].astype(np.int32)

  train_mean = x_train.mean()
  x_train -= train_mean
  x_val -= train_mean
  x_test -= train_mean

  train_std = x_train.std()
  x_train /= train_std
  x_val /= train_std
  x_test /= train_std

  return (x_train, y_train), (x_val, y_val), (x_test, y_test)
