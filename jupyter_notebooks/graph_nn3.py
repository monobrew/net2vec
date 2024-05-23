import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import os
import io

parser = argparse.ArgumentParser(description='Train the graph neural network')
parser.add_argument('--pad', help='extra padding for node embeding', type=int, default=12)
parser.add_argument('--pas', help='number of passes', type=int, default=4)
parser.add_argument('--batch_size', help='batch_size', type=int, default=64)
parser.add_argument('--lr', help='learning rate', type=float, default=0.001)
parser.add_argument('--log_dir', help='log dir', type=str, default='log')
parser.add_argument('--rn', help='number of readout neurons', type=int, default=8)
parser.add_argument('--buf', help='buffer', type=int, default=200)
parser.add_argument('-I', help='number of iteration', type=int, default=80000)
parser.add_argument('--eval', help='evaluatioin file', type=str, default='eval.tfrecords')
parser.add_argument('--train', help='train file', type=str, default='train.tfrecords')
parser.add_argument('--test', help='test file', type=str, default='test.tfrecords')
parser.add_argument('--ninf', help='Number of hidden neurions in inference layer', type=int, default=256)
parser.add_argument('--Mhid', help='Number of hidden neurons in message layer', type=int, default=8)

def stat_args(name, shift=0, scale=1):
  parser.add_argument('--{}-shift'.format(name),
                      help='Shift for {} (usualy np.mean)'.format(name),
                      type=float, default=shift)
  parser.add_argument('--{}-scale'.format(name),
                      help='Scale for {} (usualy np.std)'.format(name),
                      type=float, default=scale)

stat_args('mu', shift=0.34, scale=0.27)
stat_args('W', shift=55.3, scale=22.0)

if __name__ == '__main__':
  args = parser.parse_args()

  N_PAD = args.pad
  N_PAS = args.pas
  N_H = 2 + N_PAD
  REUSE = None
  batch_size = args.batch_size

  def M(h, e):
    with tf.name_scope('message'):
      bs = tf.shape(h)[0]
      l = tf.keras.layers.Dense(args.Mhid, activation='selu')(e)
      l = tf.keras.layers.Dense(N_H * N_H)(l)
      l = tf.reshape(l, (bs, N_H, N_H))
      m = tf.matmul(l, tf.expand_dims(h, axis=2))
      m = tf.reshape(m, (bs, N_H))
      b = tf.keras.layers.Dense(args.Mhid, activation='selu')(e)
      b = tf.keras.layers.Dense(N_H)(b)
      m = m + b
      return m

  def U(h, m, x):
    init = tf.keras.initializers.TruncatedNormal(stddev=0.01)
    with tf.name_scope('update'):
      wz = tf.Variable(name='wz', shape=(N_H, N_H), dtype=tf.float32, initializer=init)
      uz = tf.Variable(name='uz', shape=(N_H, N_H), dtype=tf.float32, initializer=init)
      wr = tf.Variable(name='wr', shape=(N_H, N_H), dtype=tf.float32, initializer=init)
      ur = tf.Variable(name='ur', shape=(N_H, N_H), dtype=tf.float32, initializer=init)
      W = tf.Variable(name='W', shape=(N_H, N_H), dtype=tf.float32, initializer=init)
      U = tf.Variable(name='U', shape=(N_H, N_H), dtype=tf.float32, initializer=init)

      z = tf.nn.sigmoid(tf.matmul(m, wz) + tf.matmul(h, uz))
      r = tf.nn.sigmoid(tf.matmul(m, wr) + tf.matmul(h, ur))
      h_tilde = tf.nn.tanh(tf.matmul(m, W) + tf.matmul(r * h, U))
      u = (1.0 - z) * h + z * h_tilde
      return u

  def R(h, x):
    with tf.name_scope('readout'):
      hx = tf.concat([h, x], axis=1)
      i = tf.keras.layers.Dense(args.rn, activation='tanh')(hx)
      i = tf.keras.layers.Dense(args.rn)(i)
      j = tf.keras.layers.Dense(args.rn, activation='selu')(h)
      j = tf.keras.layers.Dense(args.rn)(j)

      RR = tf.nn.sigmoid(i)
      RR = tf.multiply(RR, j)

      return tf.reduce_sum(RR, axis=0)

  def graph_features(x, e, first, second):
    global REUSE

    h = tf.pad(x, [[0, 0], [0, N_PAD]])
    initializer = tf.keras.initializers.glorot_uniform()

    for i in range(N_PAS):
      with tf.name_scope('features', reuse=REUSE, initializer=initializer):
        to_stack = [
          # tf.gather(x, first),  # Commented out as per the original code
          tf.gather(h, first),
          e,
          tf.gather(h, second),
          # tf.gather(x, second),  # Commented out as per the original code
        ]

        m = M(tf.gather(h, first), e)
        # Sum of messages entering a node (why does this work?)
        # num_segments as a feature is commented out as per the original code
        num_segments = tf.cast(tf.reduce_max(second) + 1, tf.int32)
        m = tf.math.unsorted_segment_sum(m, second, num_segments)
        h = U(h, m, x)

      REUSE = True

    return R(h, x)

  def inference(batch, reuse=None):
    initializer = tf.keras.initializers.glorot_uniform()
    with tf.name_scope("inference", reuse=reuse, initializer=initializer):
      l = batch
      l = tf.keras.layers.Dense(args.ninf, activation='selu')(l)
      l = tf.keras.layers.Dense(1)(l)
      return l

  def make_batch(serialized_batch):
    bs = tf.shape(serialized_batch)[0]

    to = tf.TensorArray(tf.float32, size=bs, dtype=tf.float32)
    labelto = tf.TensorArray(tf.float32, size=bs, dtype=tf.float32)

    def condition(i, a1, a2):
      return i < bs

    def body(i, to, lto):
      with tf.device("/cpu:0"):
        # Unpacking example
        with tf.name_scope('load'):
          features = tf.parse_single_example(
              serialized_batch[i],
              features={
                  'mu': tf.SparseTensor(tf.int64, tf.float32),
                  "Lambda": tf.SparseTensor(tf.int64, tf.float32),
                  "W": tf.io.FixedLenFeature([], tf.float32),
                  "R": tf.SparseTensor(tf.int64, tf.float32),
                  "first": tf.SparseTensor(tf.int64, tf.int64),
                  "second": tf.SparseTensor(tf.int64, tf.int64)
              })

          ar = [
              (
                   (tf.sparse.to_dense(features['Lambda']) - args.W_shift) / args.W_scale
          )
          ]

          x = tf.constant(ar[0][0])
          e = tf.constant(ar[0][1])

          l = graph_features(x, e, tf.sparse.to_dense(features['first']), tf.sparse.to_dense(features['second']))
          lo = tf.gather(features['R'], [0])

        to = to.write(i, l)
        lto = lto.write(i, lo)

      return i + 1, to, lto

    loop_vars = (0, to, labelto)
    i, to, lto = tf.while_loop(condition, body, loop_vars)

    to = to.stack()
    lto = lto.stack()

    to = tf.reshape(to, [bs, -1])

    pred = inference(to)

    return to, lto, pred

  def model(features, labels):
    with tf.name_scope('model'):
      x, l, pred = make_batch(features)
      loss = tf.losses.mean_squared_error(labels, pred)

      train_op = tf.train.Adam(learning_rate=args.lr).minimize(loss)

    return pred, loss, train_op

def load_data(filename):
  """
  Loads data from a TFRecords file.

  Args:
    filename: Path to the TFRecords file.

  Returns:
    A tuple of (features, labels).
  """
  dataset = tf.data.TFRecordDataset(filename)
  features, labels = dataset.map(make_batch)
  return features, labels

def train(features, labels, epochs=10, log_dir='log'):
  """
  Trains the model on the provided data.

  Args:
    features: A dataset of features.
    labels: A dataset of labels.
    epochs: Number of training epochs (default: 10).
    log_dir: Directory for logging training information (default: 'log').
  """
  pred, loss, train_op = model(features, labels)

  summary_writer = tf.summary.create_file_writer(log_dir)
  with tf.summary.record_if(True):
    tf.summary.scalar('loss', loss, step=tf.train.get_or_create_global_step())

  for epoch in range(epochs):
    for features_batch, labels_batch in tf.data.enumerate_epochs(features):
      with tf.GradientTape() as tape:
        pred_batch = model(features_batch, labels_batch)[0]
        loss_val = tf.reduce_mean(tf.losses.mean_squared_error(labels_batch, pred_batch))
      grads = tape.gradient(loss_val, model.trainable_variables)
      train_op(grads)

      tf.summary.scalar('batch_loss', loss_val, step=tf.train.get_or_create_global_step())
      summary_writer.flush()

      print(f"Epoch: {epoch+1}, Batch: {features_batch.shape[0]}, Loss: {loss_val.numpy()}")

def evaluate(features, labels):
  """
  Evaluates the model on the provided data.

  Args:
    features: A dataset of features.
    labels: A dataset of labels.

  Returns:
    The mean squared error on the evaluation data.
  """
  pred, loss, _ = model(features, labels)
  return tf.reduce_mean(tf.losses.mean_squared_error(labels, pred))

if __name__ == '__main__':
  args = parser.parse_args()

  # Load training and evaluation data
  train_features, train_labels = load_data(args.train)
  eval_features, eval_labels = load_data(args.eval)
  test_features, test_labels = load_data(args.test)  # Optional for testing

  # Train the model
  train(train_features, train_labels, epochs=args.I, log_dir=args.log_dir)

  # Evaluate the model on the evaluation data
  eval_loss = evaluate(eval_features, eval_labels)
  print(f"Evaluation Loss: {eval_loss.numpy()}")

  # Optional: Evaluate the model on the testing data
  test_loss = evaluate(test_features, test_labels)
  print(f"Testing Loss: {test_loss.numpy()}")

