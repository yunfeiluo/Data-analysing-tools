import numpy as np
import matplotlib.pyplot as plt
import model_train as mt

# ---------------------------------------------------------------- 
# Machine Learning Homework 3
#
# File_name: kernel_reg
# Functionality: all the functions for my Handwrite kernel ridge regression
# Author: Yunfei Luo
# Start date: EST Mar.29th.2020
# Last update: EST Mar.29th.2020
# ----------------------------------------------------------------


def poly_k(x1, x2, i):
        return (1+np.dot(x1, x2)) ** i

def poly_BE(x, i):
        def choose(n, k):
                mol = n
                denom = k if k != 0 else 1
                for i in range(1, k):
                        mol *= (n-i)
                        if k != 0:
                                denom *= (k-i)
                return mol / denom
        w = np.array([choose(i, k) for k in range(i+1)]) ** 0.5
        return np.array([w[k] * (x**k) for k in range(i+1)])

def trig_k(x1, x2, i):
        delta = 0.5
        res = np.dot(np.sin(delta*x1), np.sin(delta*x2)) + np.dot(np.cos(delta*x1), np.cos(delta*x2))
        for k in range(2, i+1):
                res += np.dot(np.sin(k*delta*x1), np.sin(k*delta*x2)) + np.dot(np.cos(k*delta*x1), np.cos(k*delta*x2))
        return 1 + res

def trig_BE(x, i):
        delta = 0.5
        res = np.array([1])
        for k in range(1, i+1):
                res = np.append(res, [np.sin(k*delta*x), np.cos(k*delta*x)])
        return res

def KRRS(train_x, train_y, test_x, kernel, p):
        # TODO
        K = np.zeros((len(train_x), len(train_x)))
        for i in range(len(train_x)):
                for j in range(len(train_x)):
                        K[i][j] = kernel(train_x[i], train_x[j], p)
                        if i == j:
                                K[i][j] += 0.1
        alpha = np.dot(np.linalg.inv(K), train_y)
        y_pred = np.zeros(len(test_x))
        for i in range(len(test_x)):
                y_pred[i] = sum([alpha[n]*kernel(train_x[n], test_x[i], p) for n in range(len(alpha))])
        return y_pred

def BERR(train_x, train_y, test_x, expan, p):
        # check for the dimension of new space
        new_dim = None
        if expan == poly_BE:
                new_dim = p+1
        elif expan == trig_BE:
                new_dim = 1 + 2 * p
        else:
                print('The requested expansion method does not exist!')
                exit()

        # form new dataset
        new_x = np.zeros((len(train_x), new_dim))
        for i in range(len(train_x)):
                new_x[i] = expan(train_x[i], p)

        # train model
        models = mt.models(new_x, train_y)
        model = models.train_linear_ridge(0.1, new_x, train_y)

        # make prediction
        new_test_x = np.zeros((len(test_x), new_dim))
        for i in range(len(test_x)):
                new_test_x[i] = expan(test_x[i], p)

        return model.predict(new_test_x)

# report 2.d.2
def test_all(poly_p, trig_p, train_x, train_y, test_x, test_y, compute_MSE):
        print('polynomial orders')
        for p in poly_p:
                y_pred = KRRS(train_x, train_y, test_x, poly_k, p)
                print('KRRS, order: ' + str(p) + ', MSE: ' + str(compute_MSE(test_y, y_pred)))
                y_pred = BERR(train_x, train_y, test_x, poly_BE, p)
                print('BERR, order: ' + str(p) + ', MSE: ' + str(compute_MSE(test_y, y_pred)))
                print(' ')

        print('trigonometric orders')
        for p in trig_p:
                y_pred = KRRS(train_x, train_y, test_x, trig_k, p)
                print('KRRS, order: ' + str(p) + ', MSE: ' + str(compute_MSE(test_y, y_pred)))
                y_pred = BERR(train_x, train_y, test_x, trig_BE, p)
                print('BERR, order: ' + str(p) + ', MSE: ' + str(compute_MSE(test_y, y_pred)))
                print(' ')

########## plot report 2, d, 1 ############
def plot_some(train_x, train_y, test_x, test_y):
        fig, axs = plt.subplots(2, 4)

        curr_p = [2, 6, 5, 10]

        col = 0
        method = 'poly'
        kernel = poly_k
        expan = poly_BE
        for i in range(len(curr_p)):
                if i == 2:
                        kernel = trig_k
                        expan = trig_BE
                        method = 'trig'
                y_pred = KRRS(train_x, train_y, test_x, kernel, curr_p[i])
                axs[0, col].scatter(test_x, test_y, marker='*', c='b')
                axs[0, col].scatter(test_x, y_pred, marker='o', c='r')
                axs[0, col].set_title('KRRS, ' + method + ' order, lambda=0.1, degree=' + str(curr_p[i]))

                y_pred = BERR(train_x, train_y, test_x, expan, curr_p[i])
                axs[1, col].scatter(test_x, test_y, marker='*', c='b')
                axs[1, col].scatter(test_x, y_pred, marker='o', c='r')
                axs[1, col].set_title('BERR, ' + method + ' expansion, lambda=0.1, degree=' + str(curr_p[i]))

                col += 1

        for ax in axs.flat:
                ax.set(xlabel='Test X', ylabel='True/Predicted Y')
        plt.show()