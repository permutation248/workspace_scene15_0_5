import numpy as np
import random
import logging
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


def normalize(x):
    x = (x-np.tile(np.min(x, axis=0), (x.shape[0], 1))) / np.tile((np.max(x, axis=0)-np.min(x, axis=0)), (x.shape[0], 1))
    return x



def TT_split(n_all, test_prop, seed):
    '''
    split data into training, testing dataset
    '''
    random.seed(seed)
    random_idx = random.sample(range(n_all), n_all)
    # print('random_idx:')
    # print(random_idx.shape)
  
    train_num = np.ceil((1-test_prop) * n_all).astype(np.int32)
    train_idx = random_idx[0:train_num]
    # test_num = np.floor(test_prop * n_all).astype(np.int64)
    test_num = np.floor(test_prop * n_all).astype(int)

    # print(test_prop)
    # print(train_num)
    # print(test_num)
    # exit()


    test_idx = random_idx[-test_num:]
    return train_idx, test_idx


