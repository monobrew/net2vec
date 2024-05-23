import tensorflow as tf
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import datetime
import argparse
import os

import graph_nn


args = graph_nn.args

def make_set():
    #filename_queue = tf.compat.v1.train.string_input_producer( ['test.tf.compat.v1records'])
    #reader = tf.compat.v1.tf.compat.v1RecordReader()
    #_, serialized_example = reader.read(filename_queue)
    #serialized_batch= tf.compat.v1.train.batch( [serialized_example], batch_size=200)
    ds = tf.compat.v1.data.TFRecordDataset([args.eval])
    ds = ds.batch(args.batch_size)
    serialized_batch = ds.make_one_shot_iterator().get_next()
    return serialized_batch


def main():
    REUSE=None
    g=tf.compat.v1.Graph()

    with g.as_default():
        global_step = tf.compat.v1.train.get_or_create_global_step()
        with tf.compat.v1.variable_scope('model'):
            serialized_batch = make_set()
            batch, labels = graph_nn.make_batch(serialized_batch)
            n_batch = tf.compat.v1.layers.batch_normalization(batch) 
            predictions = graph_nn.inference(n_batch)

        loss= tf.compat.v1.losses.mean_squared_error(labels,predictions)        
        
        saver = tf.compat.v1.train.Saver(tf.compat.v1.trainable_variables() + [global_step])

    with tf.compat.v1.Session(graph=g) as ses:
        ses.run(tf.compat.v1.local_variables_initializer())
        ses.run(tf.compat.v1.global_variables_initializer())

        ckpt=tf.compat.v1.train.latest_checkpoint(args.log_dir)
        if ckpt:
            print("Loading checkpint: %s" % (ckpt))
            tf.compat.v1.logging.info("Loading checkpint: %s" % (ckpt))
            saver.restore(ses, ckpt)
        
        label_py=[]
        predictions_py=[]

        for i in range(16):
            val_label_py, val_predictions_py, step = ses.run( [labels,predictions, global_step] )
            label_py.append(val_label_py)
            predictions_py.append(val_predictions_py)

        label_py = np.concatenate(label_py,axis=0)
        predictions_py = np.concatenate(predictions_py,axis=0)
        print(label_py.shape)
        print('{} step: {} mse: {} R**2: {} Pearson: {}'.format(
            str(datetime.datetime.now()),
            step,
            np.mean((label_py-predictions_py)**2),
            #np.max(np.abs(test_error)),
            graph_nn.fitquality(label_py,predictions_py),
            np.corrcoef(label_py,predictions_py, rowvar=False)[0,1] ), flush=True ) 

        plt.figure()
        plt.plot(label_py,predictions_py,'.')
        graph_nn.line_1(label_py, label_py)
        plt.grid('on')
        plt.xlabel('Label')
        plt.ylabel('Prediction')
        plt.title('Evaluation at step {}'.format(step))
        fig_path = os.path.join(args.log_dir,'eval-{0:08}.png'.format(step) )
        fig_path = 'eval.pdf'.format(step)
        plt.savefig(fig_path)
        plt.close()

        plt.figure()
        plt.hist(label_py-predictions_py,50)
        fig_path = 'rez_hist.pdf'
        plt.savefig(fig_path)
        plt.close()


if __name__ == '__main__':
    main()