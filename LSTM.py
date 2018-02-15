import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import tensorflow.contrib.layers as tf_layer

def LSTM_cell( lstm_size, keep_prob, lstm_layers, batch_size, input_x, input_y, learning_reae ):
    with tf.name_scope( 'LSTM' ):
        with tf.name_scope( 'lstm' ):
            # Baisic LSTM cell
            lstm = rnn.BasicLSTMCell( lstm_size )

        with tf.name_scope( 'drop' ):
            # Add dropout to the cell
            drop = rnn.DropoutWrapper( lstm, output_keep_prob = keep_prob )

        with tf.name_scope( 'cell' ):
            # Stack up multiple LSTM leayers, for deep learning
            cell = rnn.MultiRNNCell( [drop] * lstm_layers )

        # Stack up multiple LSTM layers, for deep learning
        initial_state = cell.zero_state( batch_size, tf.float32 )

        outputs, final_state = tf.nn.dynamic_rnn( cell, input_x, initial_state = initial_state )

        predictions = tf_layer.fully_connected( outputs[: - 1], 1 )

        cost = tf.losses.mean_squared_error( input_y, predictions )

        optimizer = tf.train.AdamOptimizer( learning_reae ).minimize( cost )

        return predictions, cost, optimizer, initial_state, final_state, cell
