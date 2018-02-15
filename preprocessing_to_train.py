import numpy as np

def select_data_y( data ):
    data_len = len( data )
    data_y = data[1 : data_len]
    arr = np.zeros( data.shape, dtype = data.type )
    arr[: data_y.shape[1]] = data_y

    return arr

def classification_data( data_x, data_y ):
    split_idx= len( data_x ) * 0.9
    train_x, val_x = data_x[: split_idx], data_x[split_idx :]
    train_y, val_y = data_y[: split_idx], data_y[split_idx :]

    # split 0
    val_x = val_x[: len( val_x ) - 1]
    val_y = val_y[: len( val_y ) - 1]

    return train_x, train_y, val_x, val_y