import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
import math
from scipy.special import comb
from math import sqrt

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
# Compute MSE
def compute_MSE(y, y_hat):
        # mean squared error
        return np.mean(np.power(y - y_hat, 2))
################################################################
        
def polynomial_kernel(x, y, p):
    k = x *y
    k += 1
    k = k ** p
    return k

def basis_expansion_poly(x,i):
    basis = []
    for j in range(len(x)):
        row = []
        for k in range(i+1):
            row.append(sqrt(comb(i,k,exact = True)) * (x[j]**k))
        basis.append(row)
    return basis

def trig_kernel(x, y, p):
    summation = 0
    for i in range(1,p+1):
        summation += (math.sin(i*0.5*x)*math.sin(i*0.5*y)) + (math.cos(i*0.5*x)*math.cos(i*0.5*y))
    summation += 1
    return summation

def basis_expansion_trig(x,i):
    basis = []
    for j in range(len(x)):
        row = []
        row.append(1)
        for k in range(1,i+1):
            row.append(math.sin(k*0.5*x[j]))
            row.append(math.cos(k*0.5*x[j]))
        basis.append(row)
    return basis

def main():
    #load data
    train_x, train_y, test_x, test_y = load_synthetic_data()
    
    poly_powers = [1,2,4,6]
    big_pred = []
    #looping through all powers of polynomial basis expansion
    for z in range(len(poly_powers)):
        kernel = []
        #computing kernel
        for i in range(len(train_x)):
            row = []
            for j in range(len(train_x)):
                row.append(polynomial_kernel(train_x[i],train_x[j],poly_powers[z]))
            kernel.append(row)
        alpha = np.linalg.inv(kernel + 0.1*np.identity(200))@train_y
        #calculating prediction
        predictions = []
        for x in range(len(test_x)):
            pred = 0
            for i in range(len(kernel)):
                pred += alpha[i]*polynomial_kernel(train_x[i],test_x[x],poly_powers[z])
            predictions.append(pred)
        #save predictions if power = 4 or 6    
        if(z == 1) or (z == 3):
            big_pred.append(predictions)
        #printing MSE
        print('KRRS, Polynomial, Degree = ' + str(poly_powers[z]) + ', MSE:')
        print(compute_MSE(test_y,predictions))
    
    #looping through all powers of polynomial kernel
    for z in range(len(poly_powers)):
        #getting basis expanded train_x vector
        new_x = basis_expansion_poly(train_x,poly_powers[z])
        #fitting sklearn ridge regression model
        ridge = Ridge(alpha=0.1)
        ridge.fit(new_x,train_y)
        #getting basis expanded test_x vector
        new_test = basis_expansion_poly(test_x,poly_powers[z])
        #calculating predictions
        predictions = ridge.predict(new_test)
        #save predictions if power = 4 or 6    
        if(z == 1) or (z == 3):
            big_pred.append(predictions)
        #printing MSE
        print('BERR, Polynomial, Degree = ' + str(poly_powers[z]) + ', MSE:')
        print(compute_MSE(test_y,predictions))
        
    trig_powers = [3,5,10]
    
    #looping through all powers of Trigonometric basis expansion
    for z in range(len(trig_powers)):
        kernel = []
        #computing kernel
        for i in range(len(train_x)):
            row = []
            for j in range(len(train_x)):
                row.append(trig_kernel(train_x[i],train_x[j],trig_powers[z]))
            kernel.append(row)
        alpha = np.linalg.inv(kernel + 0.1*np.identity(200))@train_y
        #calculating prediction
        predictions = []
        for x in range(len(test_x)):
            pred = 0
            for i in range(len(kernel)):
                pred += alpha[i]*trig_kernel(train_x[i],test_x[x],trig_powers[z])
            predictions.append(pred)
        #save predictions if power = 5 or 10    
        if(z == 1) or (z == 2):
            big_pred.append(predictions)
        #printing MSE
        print('KRRS, Trigonometric, Degree = ' + str(trig_powers[z]) + ', MSE:')
        print(compute_MSE(test_y,predictions))
    
    #looping through all powers of Trigonometric kernel
    for z in range(len(trig_powers)):
        #getting basis expanded train_x vector
        new_x = basis_expansion_trig(train_x,trig_powers[z])
        #fitting sklearn ridge regression model
        ridge = Ridge(alpha=0.1)
        ridge.fit(new_x,train_y)
        #getting basis expanded test_x vector
        new_test = basis_expansion_trig(test_x,trig_powers[z])
        #calculating predictions
        predictions = ridge.predict(new_test)
        #save predictions if power = 5 or 10    
        if(z == 1) or (z == 2):
            big_pred.append(predictions)
        #printing MSE
        print('BERR, Trigonometric, Degree = ' + str(trig_powers[z]) + ', MSE:')
        print(compute_MSE(test_y,predictions))
        
    
    #plotting all the graphs
    fig, axs = plt.subplots(4, 2,figsize=(10,20))
    
    axs[0, 0].scatter(test_x,test_y, s=10, c='b', marker="*", label='Test Samples')
    axs[0, 0].scatter(test_x,big_pred[0], s=10, c='r', marker="o", label='Predictions')
    axs[0, 0].set_title('KRRS, Polynomial, Degree = 2, Lamda = 0.1')
    
    axs[0, 1].scatter(test_x,test_y, s=10, c='b', marker="*", label='Test Samples')
    axs[0, 1].scatter(test_x,big_pred[1], s=10, c='r', marker="o", label='Predictions')
    axs[0, 1].set_title('KRRS, Polynomial, Degree = 6, Lamda = 0.1')
    
    axs[1, 0].scatter(test_x,test_y, s=10, c='b', marker="*", label='Test Samples')
    axs[1, 0].scatter(test_x,big_pred[2], s=10, c='r', marker="o", label='Predictions')
    axs[1, 0].set_title('BERR, Polynomial, Degree = 2, Lamda = 0.1')
    
    axs[1, 1].scatter(test_x,test_y, s=10, c='b', marker="*", label='Test Samples')
    axs[1, 1].scatter(test_x,big_pred[3], s=10, c='r', marker="o", label='Predictions')
    axs[1, 1].set_title('BERR, Polynomial, Degree = 6, Lamda = 0.1')
    
    axs[2, 0].scatter(test_x,test_y, s=10, c='b', marker="*", label='Test Samples')
    axs[2, 0].scatter(test_x,big_pred[4], s=10, c='r', marker="o", label='Predictions')
    axs[2, 0].set_title('KRRS, Trigonometric, Degree = 5, Lamda = 0.1')
    
    axs[2, 1].scatter(test_x,test_y, s=10, c='b', marker="*", label='Test Samples')
    axs[2, 1].scatter(test_x,big_pred[5], s=10, c='r', marker="o", label='Predictions')
    axs[2, 1].set_title('KRRS, Trigonometric, Degree = 10, Lamda = 0.1')
    
    axs[3, 0].scatter(test_x,test_y, s=10, c='b', marker="*", label='Test Samples')
    axs[3, 0].scatter(test_x,big_pred[6], s=10, c='r', marker="o", label='Predictions')
    axs[3, 0].set_title('BERR, Trigonometric, Degree = 5, Lamda = 0.1')
    
    axs[3, 1].scatter(test_x,test_y, s=10, c='b', marker="*", label='Test Samples')
    axs[3, 1].scatter(test_x,big_pred[7], s=10, c='r', marker="o", label='Predictions')
    axs[3, 1].set_title('BERR, Trigonometric, Degree = 10, Lamda = 0.1')
    
    for ax in axs.flat:
        ax.set(xlabel='X', ylabel='Y')