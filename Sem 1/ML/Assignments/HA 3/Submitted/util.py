import math
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from numpy import loadtxt, where
from pylab import scatter, show, legend, xlabel, ylabel

##implementation of sigmoid function
def Sigmoid(x):
	g = float(1.0 / float((1.0 + math.exp(-1.0*x))))
	return g

##Prediction function
def Prediction(theta, x):
	z = 0
	for i in range(len(theta)):
		z += x[i]*theta[i]
	return Sigmoid(z)


# implementation of cost functions
def Cost_Function(X,Y,theta,m):
	sumOfErrors = 0
	for i in range(m):
		xi = X[i]
		est_yi = Prediction(theta,xi)
		if Y[i] == 1:
			error = Y[i] * math.log(est_yi)
		elif Y[i] == 0:
			error = (1-Y[i]) * math.log(1-est_yi)
		sumOfErrors += error
	const = -1/m
	J = const * sumOfErrors
	#print 'cost is ', J
	return J


# gradient components called by Gradient_Descent()

def Cost_Function_Derivative(X,Y,theta,j,m,alpha):
	sumErrors = 0
	for i in range(m):
		xi = X[i]
		xij = xi[j]
		hi = Prediction(theta,X[i])
		error = (hi - Y[i])*xij
		sumErrors += error
	m = len(Y)
	constant = float(alpha)/float(m)
	J = constant * sumErrors
	return J

# execute gradient updates over thetas
def Gradient_Descent(X,Y,theta,m,alpha):
	new_theta = []
	constant = alpha/m
	for j in range(len(theta)):
		deltaF = Cost_Function_Derivative(X,Y,theta,j,m,alpha)
		new_theta_value = theta[j] - deltaF
		new_theta.append(new_theta_value)
	return new_theta

#
def func_calConfusionMatrix(predY, trueY):

	classes = np.unique(np.concatenate((trueY,predY)))
	accuracy = 0
	nOfClasses = 0
	nOfClasses += np.unique(trueY)
	nOfClasses += np.unique(predY)
	cm = np.empty((len(classes),len(classes)))
	for i,k in enumerate(classes):
		for j,l in enumerate(classes):
			cm[i,j] = np.where((trueY==k)*(predY==l))[0].shape[0]

	for i in range(len(predY)):
		if predY[i] == trueY[i]:
			accuracy += 1

	finalAccuracy = accuracy/len(predY)
	cm = np.array(cm)
	truePos = np.diag(cm)

	precision = np.sum(truePos / np.sum(cm, axis=0))
	recall = np.sum(truePos / np.sum(cm, axis=1))

	return cm, finalAccuracy, precision, recall



	# return accuracy, precision, recall
