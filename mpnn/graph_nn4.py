import tensorflow as tf
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import datetime
import argparse
import os
import io
import keras
from keras import layers, initializers

parser = argparse.ArgumentParser(description='Train the graph neural network')
parser.add_argument('--pad', help='extra padding for node embeding',  type=int, default=12)
parser.add_argument('--pas', help='number of passes',  type=int, default=4)
parser.add_argument('--batch_size', help='batch_size',  type=int, default=64)
parser.add_argument('--lr', help='learning rate',  type=float, default=0.001)
parser.add_argument('--log_dir', help='log dir',  type=str, default='log')
parser.add_argument('--rn', help='number of readout neurons',  type=int, default=8)
parser.add_argument('--buf', help='buffer',  type=int, default=200)
parser.add_argument('-I', help='number of iteration',  type=int, default=80000)
parser.add_argument('--eval', help='evaluatioin file',  type=str, default='eval.tfrecords')
parser.add_argument('--train', help='train file',  type=str, default='train.tfrecords')
parser.add_argument('--test', help='test file',  type=str, default='test.tfrecords')
parser.add_argument('--ninf', help='Number of hidden neurions in inference layer', type=int, default=256)
parser.add_argument('--Mhid', help='Number of hidden neurons in message layer', type=int, default=8)

def stat_args(name, shift=0,scale=1):
    parser.add_argument('--{}-shift'.format(name), 
        help='Shift  for {} (usualy np.mean)'.format(name) ,  
        type=float, default=shift)

    parser.add_argument('--{}-scale'.format(name), 
        help='Scale  for {} (usualy np.std)'.format(name) ,  
        type=float, default=scale)

stat_args('mu',shift=0.34, scale=0.27)
stat_args('W',shift=55.3, scale=22.0)

if __name__ == '__main__':
    args = parser.parse_args()
else:
    args = parser.parse_args([])

N_PAD=args.pad
N_PAS=args.pas
N_H=2+N_PAD
REUSE=None
batch_size=args.batch_size

def M(h,e):
    Mhid = args.Mhid
    hid_model = keras.Sequential([
        layers.Dense(Mhid, activation="selu"),
        layers.Dense(N_H * N_H)
    ])
    
    l = hid_model(e)
    l = layers.Reshape((N_H, N_H))(l)

    m = tf.matmul(l, tf.expand_dims(h, axis=-1))
    m = layers.Reshape((N_H,))(m)

    bias_model = keras.Sequential([
        layers.Dense(Mhid, activation="selu"),
        layers.Dense(N_H)
    ])

    b = bias_model(e)

    out = m + b

    return out

def U(h, m, x):
    init = initializers.TruncatedNormal(stddev = 0.01)
    with tf.name_scope('update'):
        wz = tf.get_variable(name='wz' shape=(N_H, N_H), dtype=tf.float32)