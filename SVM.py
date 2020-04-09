import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.svm import SVC
import csv

################################################################
# load corona data
def load_corona_data():
	train_x = np.load('../../Data/Corona/train_x.npy')
	train_y = np.load('../../Data/Corona/train_y.npy')
	test_x = np.load('../../Data/Corona/test_x.npy')
	return train_x, train_y, test_x
################################################################

def kaggleize(predictions,file):

	if(len(predictions.shape)==1):
		predictions.shape = [predictions.shape[0],1]

	ids = 1 + np.arange(predictions.shape[0])[None].T
	kaggle_predictions = np.hstack((ids,predictions)).astype(int)
	writer = csv.writer(open(file, 'w'))
	writer.writerow(['# id','Prediction'])
	writer.writerows(kaggle_predictions)

def main():
    print('CORONA DATA')
    #load data
    train_x, train_y, test_x = load_corona_data()
    #K-Fold set up
    kf = KFold(n_splits=5,shuffle=True, random_state=69)
    models = []
    #loop through all parameters
    parameters = [['rbf',0.5,1,1],['poly',0.5,1,3],['poly',0.5,1,5],['linear',0.5,1,1],['rbf',0.5,0.01,1],['poly',0.5,0.01,3],['poly',0.5,0.01,5],['rbf',0.5,0.001,1],['poly',0.5,0.001,3],['poly',0.5,0.001,5],['rbf',0.05,1,1],['poly',0.05,1,3],['poly',0.05,1,5],['linear',0.05,1,1],['rbf',0.05,0.01,1],['poly',0.05,0.01,3],['poly',0.05,0.01,5],['rbf',0.05,0.001,1],['poly',0.05,0.001,3],['poly',0.05,0.001,5],['rbf',0.0005,1,1],['poly',0.0005,1,3],['poly',0.0005,1,5],['linear',0.0005,1,1],['rbf', 0.0005,0.01,1],['poly', 0.0005,0.01,3],['poly', 0.0005,0.01,5],['rbf', 0.0005,0.001,1],['poly', 0.0005,0.001,3],['poly', 0.0005,0.001,5]]
    
    for i in range(len(parameters)):
        svm = SVC(kernel=parameters[i][0],C=parameters[i][1],gamma=parameters[i][2],degree=parameters[i][3])
        accuracy = []
        #K-Fold Cross Validation
        for train_index, test_index in kf.split(train_x):
            x_train, x_test = train_x[train_index], train_x[test_index]
            y_train, y_test = train_y[train_index], train_y[test_index]
            #Fit model on each fold
            svm.fit(x_train, y_train)
            #Predict on testing fold
            predict = svm.predict(x_test)
            predict = predict.reshape(len(predict),1)
            #Accuracy between prediction and true value for each K-Fold Validation
            accuracy.append(accuracy_score(y_test, predict))
        mean = np.average(accuracy)
        if(parameters[i][0] == 'rbf'):
            print("Kernel = " + parameters[i][0] + ", C = " + str(parameters[i][1]) + ", Gamma = " + str(parameters[i][2]))
            print("5-Fold Cross Validation Accuracy = " + str(mean))
        elif(parameters[i][0] == 'poly'):
            print("Kernel = " + parameters[i][0] + ", C = " + str(parameters[i][1]) + ", Gamma = " + str(parameters[i][2]) + ", Degree = " + str(parameters[i][3]))
            print("5-Fold Cross Validation Accuracy = " + str(mean))
        else:
            print("Kernel = " + parameters[i][0] + ", C = " + str(parameters[i][1]))
            print("5-Fold Cross Validation Accuracy = " + str(mean))
        models.append(mean)
        
    #final model
    print('Best Performing Model: Kernel = linear, C = 0.5')
    final = SVC(kernel = 'linear', C = 0.5)
    final.fit(train_x, train_y)
    #x_train accuracy
    y_1 = final.predict(train_x)
    y_1 = y_1.reshape(len(y_1),1)
    print('x_train accuracy:')
    print(accuracy_score(train_y, y_1))
    #saving predictions
    kaggleize(final.predict(test_x),'../Predictions/Corona/best.csv')

    del train_x, train_y, test_x, kf, parameters, mean, models, final, y_1