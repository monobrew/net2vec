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

tf1 = tf.compat.v1

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

def test():
    return args.I

N_PAD=args.pad
N_PAS=args.pas
N_H=2+N_PAD
REUSE=None
batch_size=args.batch_size

#tf.enable_eager_execution()

def parse(serialized):
    with tf.device("/cpu:0"):
        with tf.name_scope('parse'):
            features = tf1.parse_single_example(
                serialized,
                features={
                    'mu': tf1.VarLenFeature(tf.float32),
                    "Lambda": tf1.VarLenFeature( tf.float32),
                    "W":tf1.FixedLenFeature([],tf.float32),
                    "R":tf1.VarLenFeature(tf.float32),
                    "first":tf1.VarLenFeature(tf.int64),
                    "second":tf1.VarLenFeature(tf.int64)})

            ar=[(tf1.sparse_tensor_to_dense(features['mu'])-args.mu_shift)/args.mu_scale,
                    (tf1.sparse_tensor_to_dense(features['Lambda']))]
            x=tf.stack(ar,axis=1)

            e=tf1.sparse_tensor_to_dense(features['R'])
            # cecha jest od 0-1
            #e = (tf.expand_dims(e,axis=1)-0.24)/0.09
            e = tf.expand_dims(e,axis=1)

            first=tf1.sparse_tensor_to_dense(features['first'])
            second=tf1.sparse_tensor_to_dense(features['second'])
            
            W = (features['W']-args.W_shift)/args.W_scale
            
            return ((x,e,first,second),W)

def cummax(alist, extractor):
    with tf.name_scope('cummax'):
        maxes = [tf.reduce_max( extractor(v) ) + 1 for v in alist ]
        cummaxes = [tf.zeros_like(maxes[0])]
        for i in range(len(maxes)-1):
            cummaxes.append( tf.math.add_n(maxes[0:i+1]))
        
    return cummaxes

def transformation_func(it, batch_size=4):
    with tf.name_scope("transformation_func"):
        vs = [it.get_next() for _ in range(batch_size)]
        
        first_offset = cummax(vs,lambda v:v[0][2] )
        second_offset = cummax(vs,lambda v:v[0][3] )
        
    
    return ((tf.concat([v[0][0] for v in vs], axis=0),
           tf.concat([v[0][1] for v in vs], axis=0),
           tf.concat([v[0][2] + m for v,m in zip(vs, first_offset) ], axis=0),
           tf.concat([v[0][3] + m for v,m in zip(vs, second_offset) ], axis=0),
           tf.concat([ tf.cast( tf.zeros_like(vs[i][0][0][:,0]) + i, tf.int32) for i in range(batch_size) ], axis=0) ),
            tf.expand_dims(tf.stack([v[1] for v in vs], axis=0), axis=[1])
           )

def make_set():
    ds = tf.data.TFRecordDataset([args.eval])
    ds = ds.map(parse)
    ds = ds.apply(tf.data.experimental.shuffle_and_repeat(args.buf))
    it = tf1.data.make_one_shot_iterator(ds)
    with tf.device("/cpu:0"):
        return transformation_func(it, args.batch_size)


def make_trainset():
    ds = tf.data.TFRecordDataset([args.train])
    ds = ds.map(parse)
    ds = ds.apply(tf.data.experimental.shuffle_and_repeat(args.buf))
    it = tf1.data.make_one_shot_iterator(ds)
    with tf.device("/cpu:0"):
        return transformation_func(it, args.batch_size)

def make_testset():
    ds = tf.data.TFRecordDataset([args.test])
    ds = ds.map(parse)
    it = tf1.data.make_one_shot_iterator(ds)
    with tf.device("/cpu:0"):
        return transformation_func(it, args.batch_size)

def line_1(x1,x2):
    xmin=np.min(x1.tolist()+x2.tolist())
    xmax=np.max(x1.tolist()+x2.tolist())
    lines = plt.plot([1.1*xmin,1.1*xmax],[1.1*xmin,1.1*xmax])
    return lines

def fitquality (y,f):
    '''
    Computes $R^2$
    Args:
        x true label
        f predictions
    '''
    #r = np.corrcoef(np.squeeze(y),np.squeeze(f))
    #return r[0,1]
    #R2 = 1-np.var(f-y)/np.var(y)
    ssres=np.sum((y-f)**2)
    sstot=np.sum( (y-np.mean(y))**2 )
    R2 = 1-ssres/sstot

    return R2

class MessagePassing(keras.Model):
    def __init__(self):
        super(MessagePassing, self).__init__()
        
        self.l = keras.Sequential([
            keras.layers.Dense(args.Mhid,activation=tf.nn.selu),
            keras.layers.Dense(N_H*N_H),
            keras.layers.Reshape((N_H,N_H))
        ])
        
        self.b = keras.Sequential([
            keras.layers.Dense(args.Mhid,activation=tf.nn.selu),
            keras.layers.Dense(N_H)
        ])
        
        self.u = keras.layers.GRUCell(N_H)
    
    def build(self, input_shape=None):
        del input_shape
        self.l.build(tf.TensorShape([None, 1]))
        self.b.build(tf.TensorShape([None, 1]))
        self.u.build(tf.TensorShape([None, N_H]))
        self.built = True
        
    def call(self, inputs, training=False):
        (x,e,first,second,segment) = inputs
        h=tf.pad(x,[[0,0],[0,N_PAD]])
        
        for i in range(N_PAS):
            m = self._M(tf.gather(h,first),e)
            num_segments=tf.cast(tf.reduce_max(second)+1,tf.int32)
            m = tf1.unsorted_segment_sum(m,second,num_segments)
            h,_ = self.u(m,[h])
        node_batch = self._R(h,x,segment)
        f = keras.layers.Dense(args.ninf, activation='selu')(node_batch)
        f = keras.layers.Dense(1)(f)
        return f
    
    def _M(self,h,e):
        a = self.l(e)
        m=tf.matmul(a,tf.expand_dims(h,axis=2) )
        m = tf.squeeze(m)
        b = self.b(e)
        return m + b
    
    def _R(self,h,x,segment):
        hx=tf.concat([h,x],axis=1)
        i = keras.layers.Dense(args.rn, activation='tanh')(hx)
        RR = keras.layers.Dense(args.rn, activation='sigmoid')(i)
        j = keras.layers.Dense(args.rn, activation='selu')(hx)
        j = keras.layers.Dense(args.rn)(j)
        RR = tf.multiply(RR, j)
        return tf1.segment_sum(RR,segment)

if __name__ == "__main__":
    
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    print(args)

    g=tf.Graph()

    with g.as_default():

        model = MessagePassing()

        (train_batch, train_labels) = make_trainset()
        train_batch = list(train_batch)

        model.compile(
            loss = keras.losses.mean_squared_error,
            metrics = ['mae'],
            optimizer = keras.optimizers.RMSprop(learning_rate=0.001)
        )

        test_batch, test_labels = make_testset()
    
    model.summary()
    model.fit(train_batch, train_labels)



   
