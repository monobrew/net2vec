import tensorflow as tf
import keras
from keras import layers

class MessageBlock(layers.Layer):
    def __init__(self, pad : int = 12, Mhid : int = 8):
        super(MessageBlock, self).__init__()
        self.N_H = pad + 2
        self.Mhid = Mhid
        self.dense11 = layers.Dense(self.Mhid, activation="selu", kernel_initializer="zeros")
        self.dense12 = layers.Dense(self.N_H * self.N_H, kernel_initializer="zeros")
        self.reshape1 = layers.Reshape((self.N_H, self.N_H))
        self.mul = layers.Multiply()
        self.reduce_sum = layers.Lambda(lambda x : tf.reduce_sum(x, axis=2))
        self.dense21 = layers.Dense(self.Mhid, activation="selu", kernel_initializer="zeros")
        self.dense22 = layers.Dense(self.N_H)
        self.add = layers.Add()
    def call(self, h : tf.Tensor, e : tf.Tensor):
        l = self.dense11(e)
        l = self.dense12(l)
        l = self.reshape1(l)
        m = self.mul([l, h])
        m = self.reduce_sum(m)
        k = self.dense21(e)
        b = self.dense22(k)

        out = self.add([m, b])

        return out
    
class UpdateBlock(layers.Layer):
    def __init__(self, pad : int = 12):
        self.N_H = pad + 2
        super(UpdateBlock, self).__init__()
        self.gru = layers.GRUCell(self.N_H)

    def call(self, h, m):
        x = self.gru(tf.concat(m, h))
        return x

class ReadoutBlock(layers.Layer):
    def __init__(self, rn : int):
        super(ReadoutBlock, self).__init__()
        self.rn = rn

        self.concat = layers.Concatenate()
        self.dense01 = layers.Dense(self.rn, activation='tanh')
        self.dense02 = layers.Dense(self.rn, activation='sigmoid')
        self.dense11 = layers.Dense(self.rn, activation='selu')
        self.dense12 = layers.Dense(self.rn, activation='linear')
        self.mul = layers.Multiply()
        self.reduce_sum = layers.Lambda(lambda x : tf.reduce_sum(x, axis=0))
    def call(self, h, x):
        hx = self.concat([h, x])
        i = self.dense01(hx)
        i = self.dense02(i)
        j = self.dense11(h)
        j = self.dense12(j)

        RR = self.mul([i, j])
        RR = self.reduce_sum(RR)

        return RR