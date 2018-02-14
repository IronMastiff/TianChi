import numpy as np

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