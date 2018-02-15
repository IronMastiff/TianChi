import tensorflow as tf
import os
import pandas as pd

import preprocessing_to_train
import utils
import LSTM

IGNORE_BRAND = 1
PREPROCESSING_PATH = 'preprocessing_data'
TRAIN_DATA_DIR = "./data/train_20171215.txt"
PREPROCESSING_TRAIN_DATA_DIR = "preprocessing_data/pre_train_20171215.txt"

LSTM_SIZE = 512
LSTM_LAYERS = 1
BATCH_SIZE = 10
N_STEPS = 10
LEARNING_RATE = 0.001

EPOCH = 100

if not( os.path.exists( os.path.join( PREPROCESSING_PATH ) ) ):
    os.makedirs( PREPROCESSING_PATH )

if not( os.path.exists( os.path.join( PREPROCESSING_TRAIN_DATA_DIR ) ) ):
    utils.read_file( TRAIN_DATA_DIR, PREPROCESSING_TRAIN_DATA_DIR )

if IGNORE_BRAND:
    datas = utils.get_sum_cnt( PREPROCESSING_TRAIN_DATA_DIR )
else:
    datas = utils.get_datas( PREPROCESSING_TRAIN_DATA_DIR )

datas_y = preprocessing_to_train.select_data_y( datas )

train_x, train_y, val_x, val_y = preprocessing_to_train.classification_data( data_x = datas, data_y = datas_y )

with tf.Session() as sess:

    inputs, labels, keep_prob = utils.build_input()

    predictions, cost, optimizer, initial_state, final_state, cell = LSTM.LSTM_cell( LSTM_SIZE,
                                                                                     keep_prob,
                                                                                     LSTM_LAYERS,
                                                                                     BATCH_SIZE,
                                                                                     train_x,
                                                                                     labels,
                                                                                     LEARNING_RATE
                                                                                     )

    accuracy = utils.Validation( predictions, labels )

    utils.draw_scalar( cost, 'loss' )

    utils.draw_scalar( accuracy, 'Batch accurcy' )

    merged = tf.summary.merge_all()

    writer = tf.summary.FileWriter( 'logs/', sess.graph )

    sess.run( tf.global_variables_initializer() )

    with graph.as_default():
        saver = tf.train.Saver()

    iteration = 1
    for e in range( EPOCH ):
        state = sess.run( initial_state )

        for i, ( x, y ) in enumerate( utils.get_batches( train_x, train_y, BATCH_SIZE ), 1 ):
            feed = {train_x : x,
                    train_y : y[:, None],
                    keep_prob : 0.5,
                    initial_state : state}
            loss, state, _ = sess.run( [cost, final_state, optimizer], feed_dict = feed )

            if iteration % 5 == 0:
                print( "Epoch: {} / {}" . format( e, EPOCH ),
                       "Iteration: {}" . format( iteration ),
                       "Train loss: {:.3f}" . format( loss ) )

            if iteration % 25 == 0:
                val_acc = []
                val_state = sess.run( cell.zero_state( BATCH_SIZE, tf.float32 ) )
                for x, y in utils.get_batches( val_x, val_y, batch_size = BATCH_SIZE ):
                    feed = {train_x : x,
                            train_y : y[:, None],
                            keep_prob : 1,
                            initial_state : val_state}
                    batch_acc, val_state = sess.run( [accuracy, final_state], feed_dict = feed )
                    val_acc.append( batch_acc )

                print( "Val acc: {:.3f}" . format( np.mean( val_acc ) ) )
            iteration += 1
        saver.save( sess, 'checkpoints/sentment.ckpt' )