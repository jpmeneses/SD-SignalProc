import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

class CoVar(tf.keras.layers.Layer):
    def __init__(self):
        super(CoVar, self).__init__()

    def call(self, x, training=None):
        x = keras.layers.Flatten()(x)
        x_mu = tf.reduce_mean(x,axis=0,keepdims=True)
        x_dif = keras.layers.Lambda(lambda z: tf.expand_dims(z,axis=-1))(x - x_mu)
        cov = keras.layers.Lambda(lambda a: tf.linalg.matmul(a,a,transpose_b=True))(x_dif)
        cov_res = keras.layers.Lambda(lambda z: tf.reduce_mean(z,axis=0,keepdims=True))(cov)
        return cov_res
