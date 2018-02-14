import numpy as np

def get_batches( arr, batch_size, n_steps ):
    chars_per_batch = batch_size * n_steps
    n_batches = len( arr ) // chars_per_batch    #取整运算
    arr = arr[:n_batches * chars_per_batch]
    arr = arr.reshape( ( batch_size, -1 ) )
    for n in range( 0, arr.shape[1], n_steps ):
        x = arr[:, n : n + n_steps]
        y_temp = arr[:, n + 1 : n + n_steps + 1]

        y = np.zeros( x.shape, dtype = x.dtype )
        y[:, : y_temp.shape[1]] = y_temp

        yield x, y

def build_inputs( batch_size, num_steps ):
    inputs = tf.placeholder( tf.int32, [batch_size, num_steps], name = 'inputs' )
    targets = tf.placeholder( tf.int32, [batch_size, num_steps], name = 'targets' )

    keep_prob = tf.placeholder( tf.float32, name = 'keep_prob' )

    return inputs, targets, keep_prob