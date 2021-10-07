import numpy as np
from src import generate_regression_data, PolynomialRegression, mean_squared_error
try:
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

x, y = generate_regression_data(4, 100, 0.1)

train_ints10 = np.random.choice(100, 10)
train_ints50 = np.random.choice(100, 50)
x10_out = x[train_ints10]
y10_out = y[train_ints10]
x10_order = np.argsort(x10_out)
x10 = x10_out[x10_order]
y10 = y10_out[x10_order]
x90 = np.delete(x, train_ints10)
y90 = np.delete(y, train_ints10)

x50_train = x[train_ints50]
y50_train = y[train_ints50]
x50_test = np.delete(x, train_ints50)
y50_test = np.delete(y, train_ints50)

tenerrors = np.zeros(10)
tenterrors = np.zeros(10)
fiftyerrors = np.zeros(10)
fiftyterrors = np.zeros(10)

p0_10 = PolynomialRegression(0)
p0_10.fit(x10, y10)
tenerrors[0] = mean_squared_error(p0_10.predict(x90), y90)
tenterrors[0] = mean_squared_error(p0_10.predict(x10), y10)

p1_10 = PolynomialRegression(1)
p1_10.fit(x10, y10)
tenerrors[1] = mean_squared_error(p1_10.predict(x90), y90)
tenterrors[1] = mean_squared_error(p1_10.predict(x10), y10)

p2_10 = PolynomialRegression(2)
p2_10.fit(x10, y10)
tenerrors[2] = mean_squared_error(p2_10.predict(x90), y90)
tenterrors[2] = mean_squared_error(p2_10.predict(x10), y10)

p3_10 = PolynomialRegression(3)
p3_10.fit(x10, y10)
tenerrors[3] = mean_squared_error(p3_10.predict(x90), y90)
tenterrors[3] = mean_squared_error(p3_10.predict(x10), y10)

p4_10 = PolynomialRegression(4)
p4_10.fit(x10, y10)
tenerrors[4] = mean_squared_error(p4_10.predict(x90), y90)
tenterrors[4] = mean_squared_error(p4_10.predict(x10), y10)

p5_10 = PolynomialRegression(5)
p5_10.fit(x10, y10)
tenerrors[5] = mean_squared_error(p5_10.predict(x90), y90)
tenterrors[5] = mean_squared_error(p5_10.predict(x10), y10)

p6_10 = PolynomialRegression(6)
p6_10.fit(x10, y10)
tenerrors[6] = mean_squared_error(p6_10.predict(x90), y90)
tenterrors[6] = mean_squared_error(p6_10.predict(x10), y10)

p7_10 = PolynomialRegression(7)
p7_10.fit(x10, y10)
tenerrors[7] = mean_squared_error(p7_10.predict(x90), y90)
tenterrors[7] = mean_squared_error(p7_10.predict(x10), y10)

p8_10 = PolynomialRegression(8)
p8_10.fit(x10, y10)
tenerrors[8] = mean_squared_error(p8_10.predict(x90), y90)
tenterrors[8] = mean_squared_error(p8_10.predict(x10), y10)

p9_10 = PolynomialRegression(9)
p9_10.fit(x10, y10)
tenerrors[9] = mean_squared_error(p9_10.predict(x90), y90)
tenterrors[9] = mean_squared_error(p9_10.predict(x10), y10)

tens = np.array([p0_10, p1_10, p2_10, p3_10, p4_10, p5_10, p6_10, p7_10, p8_10, p9_10])

p0_50 = PolynomialRegression(0)
p0_50.fit(x50_train, y50_train)
fiftyerrors[0] = mean_squared_error(p0_50.predict(x50_test), y50_test)
fiftyterrors[0] = mean_squared_error(p0_10.predict(x50_train), y50_train)

p1_50 = PolynomialRegression(1)
p1_50.fit(x50_train, y50_train)
fiftyerrors[1] = mean_squared_error(p1_50.predict(x50_test), y50_test)
fiftyterrors[1] = mean_squared_error(p1_10.predict(x50_train), y50_train)

p2_50 = PolynomialRegression(2)
p2_50.fit(x50_train, y50_train)
fiftyerrors[2] = mean_squared_error(p2_50.predict(x50_test), y50_test)
fiftyterrors[2] = mean_squared_error(p2_10.predict(x50_train), y50_train)

p3_50 = PolynomialRegression(3)
p3_50.fit(x50_train, y50_train)
fiftyerrors[3] = mean_squared_error(p3_50.predict(x50_test), y50_test)
fiftyterrors[3] = mean_squared_error(p3_10.predict(x50_train), y50_train)

p4_50 = PolynomialRegression(4)
p4_50.fit(x50_train, y50_train)
fiftyerrors[4] = mean_squared_error(p4_50.predict(x50_test), y50_test)
fiftyterrors[4] = mean_squared_error(p4_10.predict(x50_train), y50_train)

p5_50 = PolynomialRegression(5)
p5_50.fit(x50_train, y50_train)
fiftyerrors[5] = mean_squared_error(p5_50.predict(x50_test), y50_test)
fiftyterrors[5] = mean_squared_error(p5_10.predict(x50_train), y50_train)

p6_50 = PolynomialRegression(6)
p6_50.fit(x50_train, y50_train)
fiftyerrors[6] = mean_squared_error(p6_50.predict(x50_test), y50_test)
fiftyterrors[6] = mean_squared_error(p6_10.predict(x50_train), y50_train)

p7_50 = PolynomialRegression(7)
p7_50.fit(x50_train, y50_train)
fiftyerrors[7] = mean_squared_error(p7_50.predict(x50_test), y50_test)
fiftyterrors[7] = mean_squared_error(p7_10.predict(x50_train), y50_train)

p8_50 = PolynomialRegression(8)
p8_50.fit(x50_train, y50_train)
fiftyerrors[8] = mean_squared_error(p8_50.predict(x50_test), y50_test)
fiftyterrors[8] = mean_squared_error(p8_10.predict(x50_train), y50_train)

p9_50 = PolynomialRegression(9)
p9_50.fit(x50_train, y50_train)
fiftyerrors[9] = mean_squared_error(p9_50.predict(x50_test), y50_test)
fiftyterrors[9] = mean_squared_error(p9_10.predict(x50_train), y50_train)

fifties = np.array([p0_50, p1_50, p2_50, p3_50, p4_50, p5_50, p6_50, p7_50, p8_50, p9_50])

ltenerrors = np.log(tenerrors)
lfiftyerrors = np.log(fiftyerrors)
ltenterrors = np.log(tenterrors)
lfiftyterrors = np.log(fiftyterrors)

degree = np.array([0,1,2,3,4,5,6,7,8,9])
plt.plot(degree, ltenerrors, label = "TESTING Error")
plt.plot(degree, ltenterrors, label = "TRAINING Error")
plt.title("Log Error of Regression with 10 Points of Training Data")
plt.xlabel("Degree of Polynomial")
plt.ylabel("Log Mean Squared Error")
plt.legend()
plt.savefig("Ten_Training")
plt.cla()
plt.plot(degree, lfiftyerrors, label = "TESTING Error")
plt.plot(degree, lfiftyterrors, label = "TRAINING Error")
plt.title("Log Error of Regression with 50 Points of Training Data")
plt.xlabel("Degree of Polynomial")
plt.ylabel("Log Mean Squared Error")
plt.legend()
plt.savefig("Fifty_Training")

plt.cla()
plt.plot(x10, y10, 'bo', label = 'training data')

lowerror_test = np.argmin(tenerrors)
x_val = np.linspace(-1, 1)
poly_test = np.poly1d(np.flip(tens[lowerror_test].weights))
y_val_test = poly_test(x_val)
plt.plot(x_val, y_val_test, label = 'lowest testing error, polynomial degree {}'.format(lowerror_test))

lowerror_train = np.argmin(tenterrors)
poly_train = np.poly1d(np.flip(tens[lowerror_train].weights))
y_val_train = poly_train(x_val)
plt.plot(x_val, y_val_train, label = 'lowest training error, polynomial degree {}'.format(lowerror_train))
plt.ylim(bottom= -4, top=8)
plt.title("10 Points of Training Data - Best Polynomials")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.savefig("1B_10Training")

plt.cla()
plt.plot(x50_train, y50_train, 'bo', label = 'training data')

lowerror_test = np.argmin(fiftyerrors)
x_val = np.linspace(-1, 1)
poly_test = np.poly1d(np.flip(fifties[lowerror_test].weights))
y_val_test = poly_test(x_val)
plt.plot(x_val, y_val_test, label = 'lowest testing error, polynomial degree {}'.format(lowerror_test))

lowerror_train = np.argmin(fiftyterrors)
poly_train = np.poly1d(np.flip(fifties[lowerror_train].weights))
y_val_train = poly_train(x_val)
plt.plot(x_val, y_val_train, label = 'lowest training error, polynomial degree {}'.format(lowerror_train))
plt.title("50 Points of Training Data - Best Polynomials")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.savefig("1B_50Training")