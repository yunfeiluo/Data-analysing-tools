from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize
import math

#function models: 
def lin_reg2d(x, b0, b1):
    return b0 + b1*x[0]

def lin_reg3d(x, b0, b1, b2):
    return b0 + b1*x[0] + b2*x[1]

def lin_reg4d(x, b0, b1, b2, b3):
    return b0 + b1*x[0] + b2*x[1] + b3*x[2]

def lin_reg(x, b):
    func = b[0]
    for i in range(0, len(x)):
        func += b[i + 1]*x[i]
    return func

#print function
def printFunc(b):
    result = str(b[0])
    for i in range(1, len(b)):
        result += " + " + str(b[i]) + "*(x" + str(i) + ")"
    print("The linear function is: " + result)

#error measurements
def mae(b, x, output):
    err = 0
    diff = output - lin_reg([x[0]], b)
    for i in range(0, len(output)):
        err += abs(diff[i])
    return err / len(output)

def mse(b, x, output):
    err = 0
    diff = output - lin_reg([x[0]], b)
    for i in range(0, len(output)):
        err += (diff[i]**2)
    return math.sqrt(err / len(output))

#Driver
def multiVarLinReg(f, x, s):
    output = x[len(x) - 1]
    x = x[:-1]
    b, b_covariance = optimize.curve_fit(f, x, output)
    printFunc(b)
    print("MAE: " + str(mae(b, x, output)))
    print("MSE: " + str(mse(b, x, output)))
    #print(b_covariance)
    
    #plot the data and the fitting curve if the dimension is 2 or 3
    if (s == "2d"):
        plt.figure()
        plt.scatter(x[0], output, label='Data')
        plt.plot(x[0], lin_reg([x[0]], b), label='Fitted function')
        plt.show()
    elif (s == "3d"):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(x[0], x[1], output, c='b', marker='o')

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        X, Y = np.meshgrid(x[0], x[1])
        Z = lin_reg([X, Y], b)
        ax.plot_surface(X, Y, Z)

        plt.show()

def plot2var(df):
    for i in range(len(df.iloc[1, :])):
        for j in range(i+1, len(df.iloc[1, :])):
            plt.figure()
            plt.xlabel(str(df.columns[i]), fontsize=18)
            plt.ylabel(str(df.columns[j]), fontsize=16)
            plt.scatter(df.iloc[:,i], df.iloc[:,j], label='Data')
            plt.show()

def plot3var(df):
    for i in range(len(df.iloc[1, :])):
        for j in range(i+1, len(df.iloc[1, :])):
            for k in range(j+1, len(df.iloc[1, :])):
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')

                ax.scatter(df.iloc[:, i], df.iloc[:, j], df.iloc[:, k], c='b', marker='o')

                ax.set_xlabel(str(df.columns[i]))
                ax.set_ylabel(str(df.columns[j]))
                ax.set_zlabel(str(df.columns[k]))

                plt.show()

def colorplot(df1, df2):
    for i in range(len(df1.iloc[1, :])):
        for j in range(i+1, len(df1.iloc[1, :])):
            plt.figure()
            plt.xlabel(str(df1.columns[i]), fontsize=18)
            plt.ylabel(str(df1.columns[j]), fontsize=16)
            plt.scatter(df1.iloc[:,i], df1.iloc[:,j], color='b', label='Data')
            plt.scatter(df2.iloc[:,i], df2.iloc[:,j], color='r',  label='Data')
            plt.show()