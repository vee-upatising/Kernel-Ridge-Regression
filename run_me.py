import numpy as np
import housing_kernel
import kernel_graph
import SVM
import extra_credit

##############################################################################################
# Read in train and test synthetic data
def load_synthetic_data():
    print('Reading synthetic data ...')
    train_x = np.loadtxt('../../Data/Synthetic/data_train.txt', delimiter = ',', dtype=float)
    train_y = np.loadtxt('../../Data/Synthetic/label_train.txt', delimiter = ',', dtype=float)
    test_x = np.loadtxt('../../Data/Synthetic/data_test.txt', delimiter = ',', dtype=float)
    test_y = np.loadtxt('../../Data/Synthetic/label_test.txt', delimiter = ',', dtype=float)
    return (train_x, train_y, test_x, test_y)
###############################################################################################

################################################################
# load corona data
def load_corona_data():
	train_x = np.load('../../Data/Corona/train_x.npy')
	train_y = np.load('../../Data/Corona/train_y.npy')
	test_x = np.load('../../Data/Corona/test_x.npy')
	return train_x, train_y, test_x
################################################################

################################################################
# load housing data
def load_housing_data():
    train_x = np.load('../../Data/Housing/train_x.npy')
    train_y = np.load('../../Data/Housing/train_y.npy')
    test_x = np.load('../../Data/Housing/test_x.npy')
    return train_x, train_y, test_x
################################################################

################################################################
# Compute MSE
def compute_MSE(y, y_hat):
    # mean squared error
    return np.mean(np.power(y - y_hat, 2))
################################################################

kernel_graph.main()

housing_kernel.main()

SVM.main()

extra_credit.main()
