import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVC
import csv

################################################################
# load housing data
def load_housing_data():
    train_x = np.load('../../Data/Housing/train_x.npy')
    train_y = np.load('../../Data/Housing/train_y.npy')
    test_x = np.load('../../Data/Housing/test_x.npy')
    return train_x, train_y, test_x
################################################################
    
################################################################
# load corona data
def load_corona_data():
	train_x = np.load('../../Data/Corona/train_x.npy')
	train_y = np.load('../../Data/Corona/train_y.npy')
	test_x = np.load('../../Data/Corona/test_x.npy')
	return train_x, train_y, test_x
################################################################
    
################################################################
# Compute MSE
def compute_MSE(y, y_hat):
        # mean squared error
        return np.mean(np.power(y - y_hat, 2))
################################################################
        
################################################################
def kaggleize(predictions,file,float_flag):

	if(len(predictions.shape)==1):
		predictions.shape = [predictions.shape[0],1]

	ids = 1 + np.arange(predictions.shape[0])[None].T
	if float_flag:
		kaggle_predictions = np.hstack((ids,predictions)).astype(float)
	else:
		kaggle_predictions = np.hstack((ids,predictions)).astype(int)

	writer = csv.writer(open(file, 'w'))
	if predictions.shape[1] == 1:
		writer.writerow(['# id','Prediction'])
	elif predictions.shape[1] == 2:
		writer.writerow(['# id','Prediction1', 'Prediction2'])
	writer.writerows(kaggle_predictions)
################################################################
def main():  
    print('EXTRA CREDIT')    
    #housing predictions    
    train_x, train_y, test_x = load_housing_data()
    
    train_y = train_y.reshape(len(train_y),1)
    
    #load neural network generated data
    data = np.load('synthethic_data.npy')
    train_x_new = data[:,:12]
    train_y_new = data[:,12:]
    #fit data
    final_1 = KernelRidge(kernel='rbf',alpha=0.005,gamma=1)
    print('Using KernelRidge(kernel=rbf,alpha=0.01,gamma=1), Training Set MSE:')
    final_1.fit(train_x_new, train_y_new)
    #training MSE
    y_1 = final_1.predict(train_x_new)
    y_1 = y_1.reshape(len(y_1),1)
    print(compute_MSE(train_y_new, y_1))
    #submission
    kaggleize(final_1.predict(test_x),'../Predictions/Housing/best_extra_credit.csv',True)
    
    #corona predictions
    train_x, train_y, test_x = load_corona_data()
    #load neural network generated data
    data = np.load('svm_data.npy')
    train_x = data[:,:24]
    train_y = data[:,24:]
    train_y = train_y.reshape(len(train_y),)
    #fit data
    final_2 = SVC(C=0.5, degree=1, gamma=1, kernel='linear')
    final_2.fit(train_x, train_y)
    #training accuracy
    y_1 = final_2.predict(train_x)
    y_1 = y_1.reshape(len(y_1),1)
    print('Using SVC(C=0.5, degree=1, gamma=1, kernel=linear),Training Set Accuracy:')
    print(accuracy_score(train_y, y_1))
    #submission
    kaggleize(final_2.predict(test_x),'../Predictions/Corona/best_extra_credit.csv',False)
    
    del train_x, train_y, data, train_x_new, train_y_new, y_1, final_1, final_2
