import numpy as np
import os

import utils

IGNORE_BRAND = 1
PREPROCESSING_PATH = 'preprocessing_data'
TRAIN_DATA_DIR = "./data/train_20171215.txt"
PREPROCESSING_TRAIN_DATA_DIR = "preprocessing_data/pre_train_20171215.txt"
SAVE_DIR = "./saver.txt"

MONTH_CNT_LOCA = 2
SESSION_CNT_LOCA = 3
YEAR_CNT_LOCA = 4

MONTH = 30
SESSION = 90
YEAR = 365

WEIGHT_1 = 0.25
WEIGHT_2 = 0.5
WEIGHT_3 = 0.25

if not( os.path.exists( os.path.join( PREPROCESSING_PATH ) ) ):
    os.makedirs( PREPROCESSING_PATH )

if not( os.path.exists( os.path.join( PREPROCESSING_TRAIN_DATA_DIR ) ) ):
    utils.read_file( TRAIN_DATA_DIR, PREPROCESSING_TRAIN_DATA_DIR )

if IGNORE_BRAND:
    datas = utils.get_sum_cnt( PREPROCESSING_TRAIN_DATA_DIR )
else:
    datas = utils.get_datas( PREPROCESSING_TRAIN_DATA_DIR )

datas = utils.new_datas( datas )

crop = utils.crop( datas, WEIGHT_1, WEIGHT_2, WEIGHT_3 )

datas = utils.caculate_cnt( datas, crop, MONTH_CNT_LOCA, MONTH )

datas = utils.caculate_cnt( datas, crop, SESSION_CNT_LOCA, SESSION )

datas = utils.caculate_cnt( datas, crop, YEAR_CNT_LOCA, YEAR )

datas = datas.astype( np.str )

print( datas )

utils.to_save_array( datas, SAVE_DIR )

