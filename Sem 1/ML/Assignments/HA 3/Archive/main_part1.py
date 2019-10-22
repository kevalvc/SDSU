import numpy as np
import matplotlib.pyplot as plt
from getDataset import getDataSet
from random import sample
from sklearn.linear_model import LogisticRegression
from util import Cost_Function, Gradient_Descent, Cost_Function_Derivative, Cost_Function, Prediction, Sigmoid, func_calConfusionMatrix


# Starting codes

# Fill in the codes between "%PLACEHOLDER#start" and "PLACEHOLDER#end"

# step 1: generate dataset that includes both positive and negative samples,
# where each sample is described with two features.
# 250 samples in total.

[X, y] = getDataSet()  # note that y contains only 1s and 0s,

# create figure for all charts to be placed on so can be viewed together
fig = plt.figure()


def func_DisplayData(dataSamplesX, dataSamplesY, chartNum, titleMessage):
    idx1 = (dataSamplesY == 0).nonzero()  # object indices for the 1st class
    idx2 = (dataSamplesY == 1).nonzero()
    ax = fig.add_subplot(1, 3, chartNum)
    # no more variables are needed
    plt.plot(dataSamplesX[idx1, 0], dataSamplesX[idx1, 1], 'r*')
    plt.plot(dataSamplesX[idx2, 0], dataSamplesX[idx2, 1], 'b*')
    # axis tight
    ax.set_xlabel('x_1')
    ax.set_ylabel('x_2')
    ax.set_title(titleMessage)


# plotting all samples
func_DisplayData(X, y, 1, 'All samples')

# number of training samples
nTrain = 120

######################PLACEHOLDER 1#start#########################
# write you own code to randomly pick up nTrain number of samples for training and use the rest for testing.
# WARNIN:

maxIndex = len(X)
# randomTrainingSamples = np.random.choice(maxIndex, nTrain, replace=False)

# print("RTS ", randomTrainingSamples)
# print("X ", len(X));
# print("Y ", len(y));

ln = maxIndex
numElems = int(0.8*ln) # number of elements required
# print(numElems)

indices = sample(range(ln),numElems)
# print("Indices ", len(indices))

trainX = X[indices] #  training samples
trainY = y[indices] # labels of training samples    nTrain X 1

# print("x ", np.shape(X))
# print("trx ", np.shape(trainX))
# print("try ", np.shape(trainY))

txArr = np.delete(range(ln), indices) # testing samples
testX = X[txArr]
tyArr = np.delete(range(ln), indices) # labels of testing samples     nTest X 1
testY = y[tyArr]

# print("tsx ", np.shape(testX))
# print("tsx ", np.shape(testX))
# print("tsy ", np.shape(testY))

####################PLACEHOLDER 1#end#########################

# plot the samples you have pickup for training, check to confirm that both negative
# and positive samples are included.
func_DisplayData(trainX, trainY, 2, 'training samples')
func_DisplayData(testX, testY, 3, 'testing samples')

# show all charts
plt.show()


#  step 2: train logistic regression models


######################PLACEHOLDER2 #start#########################
# in this placeholder you will need to train a logistic model using the training data: trainX, and trainY.
# please delete these coding lines and use the sample codes provided in the folder "codeLogit"

# USING SAMPLE METHOD
clf = LogisticRegression()
clf.fit(trainX,trainY)
coeffs = clf.coef_ # coefficients
intercept = clf.intercept_ # bias
bHat = np.hstack((np.array([intercept]), coeffs)) # model parameters

print('score Scikit learn: ', clf.score(testX,testY))

# visualize data using functions in the library pylab
pos = np.where(y == 1)
neg = np.where(y == 0)
# print("POS: ", pos[0])
# print("NEG: ", neg[0])
plt.scatter(X[pos, 0], X[pos, 1], marker='o', c='b')
plt.scatter(X[neg, 0], X[neg, 1], marker='x', c='r')
plt.xlabel('Feature 1: score 1')
plt.ylabel('Feature 2: score 2')
plt.legend(['Label:  Admitted', 'Label: Not Admitted'])
plt.show()


# USING OWN MODEL FROM SAMPLE
theta = [0,0] #initial model parameters
alpha = 0.1 # learning rates
max_iteration = 1000 # maximal iterations

m = len(y) # number of samples

for x in range(max_iteration):
	# call the functions for gradient descent method
	new_theta = Gradient_Descent(X,y,theta,m,alpha)
	theta = new_theta
	if x % 200 == 0:
		# calculate the cost function with the present theta
		Cost_Function(X,y,theta,m)
		# print('theta ', theta)
		# print('cost is ', Cost_Function(X,y,theta,m))

# logReg = LogisticRegression(fit_intercept=True, C=1e15) # create a model
# logReg.fit(trainX, trainY)# training
# coeffs = logReg.coef_ # coefficients
# intercept = logReg.intercept_ # bias
# bHat = np.hstack((np.array([intercept]), coeffs))# model parameters
######################PLACEHOLDER2 #end #########################



# step 3: Use the model to get class labels of testing samples.


######################PLACEHOLDER3 #start#########################
# codes for making prediction,
# with the learned model, apply the logistic model over testing samples
# hatProb is the probability of belonging to the class 1.
# y = 1/(1+exp(-Xb))
# yHat = 1./(1+exp( -[ones( size(X,1),1 ), X] * bHat )); ));
# WARNING: please DELETE THE FOLLOWING CODEING LINES and write your own codes for making predictions

# xHat = np.concatenate((np.ones((testX.shape[0], 1)), testX), axis=1)  # add column of 1s to left most  ->  130 X 3
# negXHat = np.negative(xHat)  # -1 multiplied by matrix -> still 130 X 3
# hatProb = 1.0 / (1.0 + np.exp(negXHat * bHat))  # variant of classification   -> 130 X 3
# print("HP: ", hatProb)
#
# # predict the class labels with a threshold
# yHat = (hatProb >= 0.5).astype(int)  # convert bool (True/False) to int (1/0)
#PLACEHOLDER#end

# Sample Prediction
xHat = np.concatenate((np.ones((testX.shape[0], 1)), testX), axis=1)  # add column of 1s to left most  ->  130 X 3
negXHat = np.negative(xHat)  # -1 multiplied by matrix -> still 130 X 3
hatProb = 1.0 / (1.0 + np.exp(negXHat * bHat))  # variant of classification   -> 130 X 3
yHat2 = (hatProb >= 0.5).astype(int)  # convert bool (True/False) to int (1/0)
# print("Yhat2", yHat2)
pred2 = np.empty(shape=[0, 1])
test2 = np.empty(shape=[0, 1])
for i in range(len(yHat2)):
    demo = yHat2[i].astype(int)
    # test2 = np.append(test2, [[demo]], axis=0)
    pred2 = np.append(pred2, [[demo[1]]], axis=0)

# Own Prediction
score = 0
summation = 0
pred = np.empty(shape=[0, 1])
# accuracy for sklearn
scikit_score = clf.score(testX,testY)
# accuracy for own model
length = len(testX)
for i in range(length):
    prediction = round(Prediction(testX[i],theta))
    answer = testY[i]

    difference = testY[i] - prediction
    squared_difference = difference**2
    summation = summation + squared_difference

    pred = np.append(pred, [[prediction]], axis=0)
    if prediction == answer:
        score += 1
# print("pred1 ", pred)
# print("test1 ", testY)
my_score = float(score) / float(length)
yHat = pred
######################PLACEHOLDER 3 #end #########################


# step 4: evaluation
# compare predictions yHat and and true labels testy to calculate average error and standard deviation
testYDiff = np.abs(yHat - testY)
avgErr = np.mean(testYDiff)
stdErr = np.std(testYDiff)

MSE = summation/len(testY)

print('Average Error: {} ({})'.format(avgErr, stdErr))
print('Mean Squared Error: ', MSE)

# Own Prediction
conf, acc, prec, recall = func_calConfusionMatrix(yHat, testY)
print("CM \n", conf)
print("Accuracy:", acc)
print("precision:", prec)
print("Recall:", recall)

# Sample Prediction
conf, acc, prec, recall = func_calConfusionMatrix(pred2, testY)
print("CM \n", conf)
print("Accuracy:", acc)
print("precision:", prec)
print("Recall:", recall)
