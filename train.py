import os
import pandas as pd

import preprocessing_to_train
import utils

IGNORE_BRAND = 1
PREPROCESSING_PATH = 'preprocessing_data'
TRAIN_DATA_DIR = "./data/train_20171215.txt"
PREPROCESSING_TRAIN_DATA_DIR = "preprocessing_data/pre_train_20171215.txt"

LSTM_SIZE = 512
LST_LAYERS = 1
BATCH_SIZE = 8
N_STEPS = 100
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

train_x, test_y = preprocessing_to_train.get_batches( datas, BATCH_SIZE, N_STEPS )

print( train_x )

