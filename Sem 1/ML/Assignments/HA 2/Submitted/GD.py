
import numpy as np
# X          - single array/vector
# y          - single array/vector
# theta      - single array/vector
# alpha      - scalar
# iterations - scarlar

def gradientDescent(X, y, theta, alpha, numIterations):
    '''
    # This function returns a tuple (theta, Cost array)
    '''
    totfincost = 0
    residualError = 0
    m = len(y)
    arrCost = [];

    transposedX = np.transpose(X) # transpose X into a vector  -> XColCount X m matrix

    for iteration in range(0, numIterations):
        ################PLACEHOLDER3 #start##########################
        #: write your codes to update theta, i.e., the parameters to estimate.
	    # Replace the following variables if needed

        # Our Prediction
        y0 = np.dot(X, theta)
        # print("y0 ", y0)

        # Error: Our prediction - the ground truth
        gdDiff = y0 - y
#        print("Res Err ", residualError)

        gradient = (1/m) * np.dot(transposedX, gdDiff)


        change = [alpha * x for x in gradient]
        theta = np.subtract(theta, change)  # or theta = theta - alpha * gradient
#        print("Theta", theta)
        ################PLACEHOLDER3 #end##########################

        ################PLACEHOLDER4 #start##########################
        # calculate the current cost with the new theta;
        # residualError = np.sum((gdDiff)**2/(2*m))

        # totfincost += np.square(transposedX-y)
        atmp = np.sum((gdDiff)**2/(2*m))


#        print(atmp)
        arrCost.append(atmp)
        # cost = (1 / m) * np.sum(residualError ** 2)
        ################PLACEHOLDER4 #end##########################

        # print(len(arrCost), ", arrcost: ", arrCost)

    return theta, arrCost
