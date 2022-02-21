import tensorflow as tf
import numpy as np
import os

from scipy.special import eval_hermitenorm


class SE2DIN(tf.keras.layers.Layer):
  """
  computes fundamental differential invariants of SE(2) on 2D images using Gaussian derivatives
  """
  def __init__(self, sigma, width, order=2, padding='SAME',
             **kwargs):
    super().__init__(**kwargs)
    self.sigma = sigma
    self.width = width
    self.padding = padding
    self.order = int(order)
    # for we are assuming channel last
    self.channel_axis = -1
    
    
    x = np.arange(-width, width+1, dtype=np.float32)
    g0 = np.exp(-(x**2)/(2*sigma**2))
    g0 /= np.sqrt(2 * np.pi * sigma)


    g = [g0]
    for n in range(1,3):
      tmp = (1-2*(n%2))*eval_hermitenorm(n, x/sigma)*g0
      tmp /= sigma ** n
      g.append(tmp)
    
    self.gx = [p.reshape([-1,1,1,1]) for p in g]
    self.gy = [p.reshape([1,-1,1,1]) for p in g]

  def build(self, input_shape):
    if input_shape[self.channel_axis] is None:
      raise ValueError('The channel dimension of the inputs '
                         'should be defined. Found `None`.')
        
    n_channels = input_shape[-1]
    self.gx = [np.concatenate(n_channels*[p], 2) for p in self.gx]
    self.gy = [np.concatenate(n_channels*[p], 2) for p in self.gy]
    self.gx = [tf.constant(p, dtype=tf.float32) for p in self.gx]
    self.gy = [tf.constant(p, dtype=tf.float32) for p in self.gy]

    # Be sure to call this at the end
    super(DISE2, self).build(input_shape)
  


  def _get_derivatives_1(self, x):
    tmp = tf.nn.depthwise_conv2d(x, self.gx[1], (1,1,1,1), self.padding)
    ux = tf.nn.depthwise_conv2d(tmp, self.gy[0], (1,1,1,1), self.padding)
    tmp = tf.nn.depthwise_conv2d(x, self.gy[1], (1,1,1,1), self.padding)
    uy = tf.nn.depthwise_conv2d(tmp, self.gx[0], (1,1,1,1), self.padding)
    return ux, uy

  def _get_derivatives(self, x):
    tmp = tf.nn.depthwise_conv2d(x, self.gx[0], (1,1,1,1), self.padding)
    tmp = tf.nn.depthwise_conv2d(tmp, self.gy[0], (1,1,1,1), self.padding)
    out = [tmp]
    for n in range(1,3):
      for i in range(0,n+1):
        tmp = tf.nn.depthwise_conv2d(x, self.gx[n-i], (1,1,1,1), self.padding)
        tmp = tf.nn.depthwise_conv2d(tmp, self.gy[i], (1,1,1,1), self.padding)
        out.append(tmp)
    return tuple(out)

  def call(self, inputs):
    u, ux, uy, uxx, uxy, uyy = self._get_derivatives(inputs)

    di1 = ux*ux + uy*uy
    N = tf.sqrt(di1)
    di2 = .5*((uxx*ux*ux) + (2*uxy*ux*uy) + (uyy*uy*uy)) / (N + 1e-10)
    di3 = .5*(ux*uy*(uyy - uxx) + uxy*(ux*ux - uy*uy)) / (N + 1e-10)
    di4 = .5*(uxx*uy*uy - 2*uxy*ux*uy + uyy*ux*ux) / (N + 1e-10)
    out = [u,di1,di2,di3,di4]
    start = 2
    for i in range(2, self.order):
      k = len(out)
      for j in range(start, k):
        v = out[j]
        vx, vy = self._get_derivatives_1(v)
        if j < k-1:
          out.append((vx*ux + vy*uy) / (N + 1e-10))
        else:
          out.append((-vx*uy + vy*ux) / (N + 1e-10))
      start += i+1


    out = tf.concat(out, 3)
    return out

  def get_config(self):
    config = super().get_config().copy()
    config['sigma'] = self.sigma
    config['width'] = self.width
    config['order'] = self.order
    config['padding'] = self.padding
    return config
