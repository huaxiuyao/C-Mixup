import numpy as np

def get_Airfoil_data_packet(args,path = './data/Airfoil/'):

    ########### input ###########
    fboj = open(path + 'airfoil_self_noise.dat')

    data = []

    for eachline in fboj:
        t=eachline.strip().split('\t')
        data.append([*map(float, t)])

    data = np.array(data)

    ########### shuffle ############

    shuffle_idx = np.random.permutation(data.shape[0])
    data = data[shuffle_idx]

    ########## x normalization ########

    x_data = data[:,0:5]
    y_data = data[:,5:]

    x_max = np.amax(x_data, axis = 0)
    x_min = np.amin(x_data, axis = 0)
    x_data = (x_data - x_min) / (x_max - x_min)

    ########## split ###########

    x_train = x_data[:1003,:]
    x_valid = x_data[1003:1303,:]
    x_test = x_data[1303:1503,:]

    y_train = y_data[:1003,:]
    y_valid = y_data[1003:1303,:]
    y_test = y_data[1303:1503,:]

    data_packet = {
        'x_train': x_train,
        'x_valid': x_valid,
        'x_test': x_test,
        'y_train': y_train,
        'y_valid': y_valid,
        'y_test': y_test,
    }

    return data_packet


if __name__ == '__main__':
    data_packet = get_Airfoil_data_packet()