import numpy as np

def get_NO2_data_packet(args,path = './data/NO2/'):
    x_train = np.load(path + 'x_train.npy')
    x_valid = np.load(path + 'x_valid.npy')
    x_test = np.load(path + 'x_test.npy')

    y_train = np.load(path + 'y_train.npy')
    y_valid = np.load(path + 'y_valid.npy')
    y_test = np.load(path + 'y_test.npy')

    data_packet = {
        'x_train': x_train,
        'x_valid': x_valid,
        'x_test': x_test,
        'y_train': y_train,
        'y_valid': y_valid,
        'y_test': y_test,
    }
    return data_packet