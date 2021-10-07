import numpy as np

def mean_squared_error(estimates, targets):
    n = estimates.shape[0]

    difference = targets - estimates
    square = np.square(difference)
    sum = np.sum(square)
    MSE = (1/n) * sum

    return MSE