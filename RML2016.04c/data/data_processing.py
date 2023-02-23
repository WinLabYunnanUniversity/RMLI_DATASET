# 作者：ruby
# 开发时间：2022/12/26 20:22
import pickle,torch,math,os,h5py,csv
import random,statistics
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
def load_data(data_dir):
    print('Data stored in %s' % data_dir)
    Xd = pickle.load(open(data_dir, 'rb'), encoding='latin')
    snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])
    X = []
    lbl = []
    for mod in mods:
        for snr in snrs:
            X.append(Xd[(mod, snr)])
            for i in range(Xd[(mod, snr)].shape[0]):  lbl.append((mod, snr))
    X = np.vstack(X)
    lbl =np.array(lbl)
    print('X:',X.shape)

    n_examples = X.shape[0]
    n_train = int(n_examples * .8)
    # random.seed(1)
    train_idx = np.random.choice(range(0, n_examples), size=n_train, replace=False)
    test_idx = list(set(range(0, n_examples)) - set(train_idx))
    idx = np.random.choice(range(0, n_examples), size=n_examples, replace=False)
    X_train,X_test = X[train_idx],X[test_idx]
    lbl_train, lbl_test = lbl[train_idx],lbl[test_idx]
    lbl_data = lbl[idx]
    # print('X_train.shape={},X_test.shape={}'.format(X_train.shape, X_test.shape))
    # print('modulation labels: {}'.format(mods))
    # separating labels
    train_labels = np.array(list(map(lambda x: mods.index(lbl[x][0]), train_idx)), dtype=int)
    test_labels = np.array(list(map(lambda x: mods.index(lbl[x][0]), test_idx)), dtype=int)
    Y = np.array(list(map(lambda x: mods.index(lbl[x][0]), idx)), dtype=int)
    # reshaping data - add one dimension
    x_train = np.expand_dims(X_train, axis=1)
    x_test = np.expand_dims(X_test, axis=1)
    X = np.expand_dims(X[idx], axis=1)
    all_data = [X, Y, lbl_data, snrs, mods]
    divide_data =[x_train, x_test, train_labels, test_labels, lbl_train, lbl_test]
    return all_data,divide_data


def dataloader(x,y,batch_size = 64):
    x = torch.Tensor(x)
    y = torch.Tensor(y)

    dataset = torch.utils.data.TensorDataset(x, y.type(torch.LongTensor))

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader



def find_snr(X,Y,lbl,dest_snr):
    # lbl_mods = np.array(list(map(lambda x: x[0], lbl)))
    lbl_snrs = np.array(list(map(lambda x: int(x[1]), lbl)))
    # print('lbl_snrs:',lbl.shape)

    X_snr= X[np.where(lbl_snrs == dest_snr)[0]]
    lbl_snr = lbl[np.where(lbl_snrs == dest_snr)[0]]
    Y_snr = Y[np.where(lbl_snrs == dest_snr)[0]]

    return X_snr,Y_snr,lbl_snr

def get_signal_noise_power(x,snr):
    snr_linear = 10**(snr/10)
    temp = list(map(lambda x1:np.linalg.norm(x1.reshape(-1))**2,x))
    x_power=np.mean(np.array(temp))
    # print('x_power:', x_power)
    n_power = x_power / (1+snr_linear)
    s_power = snr_linear *n_power
    return s_power,n_power

def get_distubution_power(x_power,dest_value):
    value_linear = 10**(dest_value/10)
    v_power = x_power*value_linear
    return v_power

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def rmlvector(data):
    data = np.squeeze(data)
    print('data.shape',data.shape)
    nums, rows, cols = data.shape
    dataVector = np.zeros((nums,rows * cols))
    dataVector = np.reshape(data, (nums,rows * cols))
    print('dataVector:',dataVector.shape)
    return dataVector

# 定义PCA算法
def PCA(data, r):
    data = np.float32(np.mat(data))
    u,s,v = np.linalg.svd(data)
    V = v.T
    V_r = V[:, 0:r].A  # 按列取前r个特征向量
    return V_r



