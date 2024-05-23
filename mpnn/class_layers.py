import tensorflow as tf
from keras import layers

class MessageLayer(layers.Layer):
    def __init__(self, args):
        super(MessageLayer, self).__init__()
        self.N_H = args.pad + 2
        self.Mhid = args.Mhid
    def call(self, h, e):
        l = layers.Dense(self.Mhid, activation="selu")(e)
        l = layers.Dense(self.N_H * self.N_H, kernel_initializer="zeros")(l)
        l = tf.reshape((self.N_H, self.N_H))(l)

        m = tf.matmul(l, tf.expand_dims(h, axis=-1))
        m = layers.Reshape((self.N_H,))(m)
        
        l = layers.Dense(self.Mhid, activation="selu", kernel_initializer="zeros")(e)
        b = layers.Dense(self.N_H)(l)

        out = m + b

        return out