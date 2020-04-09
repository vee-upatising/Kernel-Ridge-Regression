import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.kernel_ridge import KernelRidge
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
    print('HOUSING DATA')
    #load data
    train_x, train_y, test_x = load_housing_data()
    #reshape y to match predictions shape
    train_y = train_y.reshape(len(train_y),1)
    
    kf = KFold(n_splits=5,shuffle=True, random_state=69)
    models = []
    #loop through all parameters
    parameters = [['rbf',1,0.001],['poly',1,0.001],['linear',1,0.001],['rbf',1,0.01],['poly',1,0.01],['rbf',1,0.1],['poly',1,0.1],['rbf',1,1],['poly',1,1],['rbf',0.1,0.001],['poly',0.1,0.001],['linear',0.1,0.001],['rbf',0.1,0.01],['poly',0.1,0.01],['rbf',0.1,0.1],['poly',0.1,0.1],['rbf',0.1,1],['poly',0.1,1],['rbf',0.01,0.001],['poly',0.01,0.001],['linear',0.01,0.001],['rbf',0.01,0.01],['poly', 0.01,0.01],['rbf', 0.01,0.1],['poly', 0.01,0.1],['rbf', 0.01,1],['poly', 0.01,1],['rbf',0.001,0.001],['poly',0.001,0.001],['linear',0.001,0.001],['rbf',0.001,0.01],['poly', 0.001,0.01],['rbf', 0.001,0.1],['poly', 0.001,0.1],['rbf', 0.001,1],['poly', 0.001,1],['rbf', 0.005,0.001],['poly', 0.005,0.001],['linear', 0.005,0.001],['rbf', 0.005,0.01],['poly', 0.005,0.01],['rbf', 0.005,0.1],['poly', 0.005,0.1],['rbf', 0.005,1],['poly', 0.005,1]]
    
    for i in range(len(parameters)):
        kernel = KernelRidge(kernel=parameters[i][0],alpha=parameters[i][1],gamma=parameters[i][2])
        accuracy = []
        #K-Fold Cross Validation
        for train_index, test_index in kf.split(train_x):
            x_train, x_test = train_x[train_index], train_x[test_index]
            y_train, y_test = train_y[train_index], train_y[test_index]
            #Fit model on each fold
            kernel.fit(x_train, y_train)
            #Predict on testing fold
            predict = kernel.predict(x_test)
            predict = predict.reshape(len(predict),1)
            #Accuracy between prediction and true value for each K-Fold Validation
            accuracy.append(compute_MSE(y_test, predict))
        mean = np.average(accuracy)
        if(parameters[i][0] == 'rbf'):
            print("Kernel = " + parameters[i][0] + ", Alpha = " + str(parameters[i][1]) + ", Gamma = " + str(parameters[i][2]))
            print("5-Fold Cross Validation MSE = " + str(mean))
        elif(parameters[i][0] == 'poly'):
            print("Kernel = " + parameters[i][0] + ", Alpha = " + str(parameters[i][1]) + ", Gamma = " + str(parameters[i][2]))
            print("5-Fold Cross Validation MSE = " + str(mean))
        else:
            print("Kernel = " + parameters[i][0] + ", Alpha = " + str(parameters[i][1]))
            print("5-Fold Cross Validation MSE = " + str(mean))
        models.append(mean)
    
    #Best performing model
    final = KernelRidge(kernel='rbf',alpha=0.01,gamma=1)
    final.fit(train_x, train_y)
    
    print('Best Performing Model: Kernel = rbf, Alpha = 0.1, Gamma = 1')
    
    y_1 = final.predict(train_x)
    y_1 = y_1.reshape(len(y_1),1)
    y_1 = y_1.astype(int)
    #x_train accuracy
    print('X_train MSE:')
    print(compute_MSE(train_y, y_1))
    #saving predictions
    kaggleize(final.predict(test_x),'../Predictions/Housing/best.csv',True)
    
    del train_x, train_y, test_x, kf, parameters, mean, models, final, y_1