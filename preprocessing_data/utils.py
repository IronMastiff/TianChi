import numpy as np

def read_file( dir, save_dir ):
    with open( dir, 'r' ) as f:
        text = f.read()[27:]
    to_save_array( text, save_dir )

def to_save_array( data, save_dir ):
    with open( save_dir, 'w') as f:
        f.write( data )

def get_sum_cnt( dir ):
    datas = np.loadtxt( dir, dtype = int )
    array = np.zeros([1032, 2], dtype = int )
    for data in datas:
        array[data[0] - 1, 0] = data[0]
        array[data[0] - 1, 1] += data[3]
    return array

def get_datas( dir ):
    datas = np.loadtxt( dir, dtype = int )
    return datas

def new_datas( datas ):
    new_datas = np.zeros( [datas.shape[0], datas.shape[1] + 3], dtype=datas.dtype )
    new_datas[:, : datas.shape[1]] = datas
    return new_datas

def crop( datas, weight_1, weight_2, weight_3 ):
    crop = np.zeros( [datas.shape[0] - 2, 1], dtype = datas.dtype )
    for i in range( datas.shape[0] - 2 ):
        crop[i] = ( datas[i, 1] * weight_1 + datas[i + 1, 1] * weight_2 + datas[i + 2, 1] * weight_3 ) / 3
    return crop


def caculate_cnt( datas, crop, loca, date ):
    datas[date : , loca] = np.transpose( crop[: datas.shape[0] - date] )
    return datas