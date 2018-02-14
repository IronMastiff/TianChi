import time
import numpy as np
import tensorflow as tf

def build_inputs( batch_size, num_steps ):
    inputs = tf.placeholder( tf.int32, [batch_size, num_steps], name = 'inputs' )
    targets = tf.placeholder( tf.int32, [batch_size, num_steps], name = 'targets' )

    keep_prob = tf.placeholder( tf.float32, name = 'keep_prob' )

    return inputs, targets, keep_prob

def build_lstm( lstm_size, num_layers, batch_size, keep_prob ):
    def build_cell( lstm_size, keep_prob ):
        #Use a baisic LSTM cell
        lstm = tf.contrib.rnn.BasicLSTMCell( lstm_size )
        #Add dropout to the cell
        drop = tf.contrib.rnn.DropoutWrapper( lstm, output_keep_prob = keep_prob )
        return drop
    #Stack up multiple LSTM layers, for deep learning
    cell = tf.contrib.rnn.MultiRNNCell( [build_cell( lstm_size, keep_prob )] )
    initial_state = cell.zero_state( batch_size, tf.float32 )

    return cell, initial_state

def build_output( lstm_output, in_size, out_size ):
    seq_output = tf.concat( lstm_output, axis = 1 )

