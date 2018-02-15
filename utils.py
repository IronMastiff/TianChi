import numpy as np
import tensorflow as tf

def read_file( dir, save_dir ):
    with open( dir, 'r' ) as f:
        text = f.read()[27:]    #字符串分片
    to_save_array( text, save_dir )

def to_save_array( data, save_dir ):
    with open( save_dir, 'w') as f:
        f.write( data )

def get_sum_cnt( dir ):
    datas = np.loadtxt( dir, dtype = int )
    array = np.zeros([1032, 1], dtype = int )
    for data in datas:
        array[data[0] - 1, 0] += data[3]
    return array

def get_datas( dir ):
    datas = np.loadtxt( dir, dtype = int )
    return datas

def get_batches( x, y, batch_size = 100 ):
    n_batches = x // batch_size
    x, y = x[: n_batches * batch_size], y[: n_batches * batch_size]
    for i in range( 0, len( x ), batch_size ):
        yield x[i : batch_size + i], y[i : batch_size + i]

def build_input():
    with tf.name_LSTM( 'LSTM' ):
        with tf.name_scope( 'input' ):
            input_x = tf.placeholder( tf.int32, [None, None], name = 'inputs' )
            input_y = tf.placeholder( tf.int32, [None, None], name = 'labels' )
            keep_prob = tf.placeholder( tf.int32, name = "keep_prob" )

            return input_x, input_y, keep_prob

def Validation( predictions, val_y ):
    with tf.name_scope( 'Validation' ):
        correct_pred = tf.equal( tf.cast( tf.round( predictions ), tf.int32 ), val_y )
        accuracy = tf.reduce_mean( tf.cast( correct_pred, tf.float32 ) )

        return accuracy

def draw_scalar( value, neme ):
    tf.summary.scalar( value, name )

def draw_histogran( value, name ):
    tf.summary.histogram( value, name )