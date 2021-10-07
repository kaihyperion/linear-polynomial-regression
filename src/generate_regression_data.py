import numpy as np 

def generate_regression_data(degree, N, amount_of_noise=1.0):
    
    np.random.seed()
    x = np.linspace(-1, 1, num= N) #N values of x equally spaced between -1 and 1
    #x = 2 * np.random.rand(N) - 1 #N random values of x between -1 and 1
    random_coef = 20 * np.random.rand(degree + 1) - 10 #random coefficients between -10 and 10
    y = np.zeros(N)
    for x_i in range(N):
        y[x_i] += random_coef[0]
        for d in range(degree):
            y[x_i] += (random_coef[d+1] * (x[x_i]**(d+1)))

    noise = np.std(y) * amount_of_noise
    y_noise = y + np.random.normal(loc=0.0, scale=noise, size=y.shape)

    return x, y_noise
