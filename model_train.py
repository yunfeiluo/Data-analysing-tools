import numpy as np
from sklearn import neighbors, tree, linear_model, model_selection, neural_network, kernel_ridge, svm
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- 
# Machine Learning Homework 3
#
# File_name: model_train
# Functionality: the collection of models, include some model evaluation functions
# Author: Yunfei Luo
# Start date: EST Mar.29th.2020
# Last update: EST Mar.29th.2020
# ----------------------------------------------------------------

class models:
    def __init__(self, input_, output):
        self.x = input_
        self.y = output
        self.kernel = None
        self.degree = 3
        self.alpha = None
        self.gamma = None

    def setData(self, input_, output):
        self.x = input_
        self.y = output

    def train_svm_classifier(self, x_, y_):
        return svm.SVC(C=self.alpha, kernel=self.kernel, degree = self.degree, gamma=self.gamma).fit(x_, y_)

    def train_kernel_ridge_regression(self, x_, y_):
        return kernel_ridge.KernelRidge(alpha=self.alpha, kernel=self.kernel, gamma=self.gamma).fit(x_, y_)

    def train_nn(self):
        return neural_network.MLPRegressor(max_iter=10000, hidden_layer_sizes=(100,50,25)).fit(self.x, self.y)

    # decision tree regression
    def train_decision_tree_regression(self, depth, x_, y_):
        return tree.DecisionTreeRegressor(max_depth = depth).fit(x_, y_)
    
    # train decision tree classification
    def train_decision_tree_classification(self, depth, x_, y_):
        return tree.DecisionTreeClassifier(criterion="entropy", max_depth=depth).fit(x_, y_)

    # train nearest neighbor regressor
    def train_knn_regression(self, k, x_, y_):
        return neighbors.KNeighborsRegressor(n_neighbors=k).fit(x_, y_)
    
    # train nearest neighbor classifier
    def train_knn_classification(self, k, x_, y_):
        return neighbors.KNeighborsClassifier(n_neighbors=k).fit(x_, y_)

    # train linear model with l2-norm regularizor
    def train_linear_ridge(self, alpha_, x_, y_):
        return linear_model.Ridge(alpha=alpha_).fit(x_, y_)
    
    # train linear model with l1-norm regularizor
    def train_linear_lasso(self, alpha_, x_, y_):
        return linear_model.Lasso(alpha=alpha_).fit(x_, y_)
    
    # train linear model with SGD, l2-norm regularizor
    def train_linear_SGD_classification(self, alpha_, x_, y_):
        #loss = 'hinge'
        loss = 'log'
        return linear_model.SGDClassifier(loss=loss, penalty='l2', alpha=alpha_).fit(x_, y_)

    def model_eval(self, compute_error, train, k):
        kf = model_selection.KFold(n_splits=k, shuffle=True)
        err = 0
        for train_index, test_index in kf.split(self.x): # loop for cross validation
            # extract training data
            x_train = np.array([self.x[i] for i in train_index])
            y_train = np.array([self.y[i] for i in train_index])

            # extract test data
            x_test = np.array([self.x[i] for i in test_index])
            y_test = np.array([self.y[i] for i in test_index])

            # train model, make prediction
            model = train(x_train, y_train)
            y_predict = model.predict(x_test)

            # compute error
            err += compute_error(y_test, y_predict)
        return err / k

    # k-fold cross validation
    def crossVal(self, compute_error, train, k, hs):
        kf = model_selection.KFold(n_splits=k, shuffle=True)
        capacity = len(hs)
        errors = list()
        chosen = np.inf
        chosen_ind = -1
        ind = -1
        for h in hs: # loop among each hyper-parameter
            print('h = ' + str(h))
            ind += 1
            error = 0
            for train_index, test_index in kf.split(self.x): # loop for cross validation
                # extract training data
                x_train = np.array([self.x[i] for i in train_index])
                y_train = np.array([self.y[i] for i in train_index])

                # extract test data
                x_test = np.array([self.x[i] for i in test_index])
                y_test = np.array([self.y[i] for i in test_index])

                # train model, make prediction
                model = train(h, x_train, y_train)
                y_predict = model.predict(x_test)

                # compute error
                error += compute_error(y_test, y_predict)
            errors.append(error/k)
            print(errors[-1])
            if errors[ind] <= chosen:
                chosen = errors[ind]
                chosen_ind = ind

        print('best h: ' + str(hs[chosen_ind]) + ', error: ' + str(chosen))
            
        plt.bar(range(capacity), errors, tick_label=hs)
        plt.xlabel('h')
        plt.ylabel('error')
        plt.show()

        return hs[chosen_ind]
