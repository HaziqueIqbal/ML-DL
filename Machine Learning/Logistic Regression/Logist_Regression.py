import numpy as np
import math
import copy
from Utils.input_validation import check_inputs
# Linear Regression Model
class Logistic_Regression:
    # Init function
    def __init__(self, x_train, y_train, learning_rate, number_of_iterations, lambda_=1):
        # Checking for valid inputs
        if not (check_inputs(x_train, y_train, learning_rate, number_of_iterations, 2)):
            exit()
        # Extracting shape of training set
        m, n = x_train.shape
        # Initializing parameter with 0's
        w_init = np.zeros(n)
        b_init = 0.
        # Getting values from Linear Regression Model
        self.cost, self.parameters, self.w_final, self.b_final = Logistic_Regression.gradient_descent(
            x_train, y_train, w_init, b_init, learning_rate, number_of_iterations, lambda_)

    # Sigmoid function
    def sigmoid(z):
        return 1 / (1+np.exp(-z))
    
    # Cost function
    def cost_function(X, Y, w, b, lambda_=1):
        # Extracting shape of training set
        m, n = X.shape
        # Initializing costs to 0
        cost = 0
        reg_cost = 0
        for i in range(m):
            # Model
            f_wb = np.dot(X[i], w) + b
            g_z = Logistic_Regression.sigmoid(f_wb)
            cost += -Y[i] * np.log(g_z) - (1 - Y[i]) * np.log(1 - g_z)
        # Regularization cost
        for j in range(n):
            reg_cost += (w[j]**2)
        # Multiplying with lambda
        reg_cost *= (lambda_ / (2*m))
        # Dividing it by total training examples
        total_cost = cost / m
        # Adding two costs
        total_cost += reg_cost
        # Return total cost
        return total_cost

    # Calulcating the derivate of the cost function w.r.t "w" and "b" parameters
    def gradient_function(X, Y, w, b, lambda_=1):
        # Extracting shape of training set
        m, n = X.shape
        # Initializing derivative variables with 0's
        dj_dw = np.zeros(n)
        dj_db = 0.

        for i in range(m):
            # Model
            f_wb = np.dot(X[i], w) + b
            g_z = Logistic_Regression.sigmoid(f_wb)
            # error
            error = g_z - Y[i]
            for j in range(n):
                # Multiplying error with each feature
                dj_dw[j] += error * X[i][j]
            dj_db += error
        # Dividing derivative variable with total training examples
        dj_dw /= m
        dj_db /= m
        # Regularization of paramter w
        for k in range(n):
            dj_dw[k] += (lambda_ * w[k]) / m
        # Return derivative for w and b
        return dj_dw, dj_db

    # Gradient Descent
    def gradient_descent(X, Y, w_init, b_init, learning_rate, iterations, lambda_):
        # histories for storing all change in cost, w and b
        cost_history = []
        parameters_history = []
        
        # Assigning w_init and b_init to variable w and b with new reference
        w = copy.deepcopy(w_init)
        b = copy.deepcopy(b_init)

        for i in range(iterations):
            # Caluclating values for derivative
            dj_dw, dj_db = Logistic_Regression.gradient_function(
                X, Y, w, b, lambda_)

            # Chnage in variable w and b
            w = w - learning_rate * dj_dw
            b = b - learning_rate * dj_db

            # Appending change in history variables
            cost_history.append(
                Logistic_Regression.cost_function(X, Y, w, b, lambda_))
            parameters_history.append([w, b])

            # Print
            if i % math.ceil(iterations/10) == 0:
                print(f"Iteration: {i}, cost: {cost_history[-1]}")

        # Return histories and parameters
        return cost_history, parameters_history, w, b

    # Prediction function for Linear Regression
    def predict(self, X, Y, w, b):
        # Number of training examples
        m = X.shape[0]
        # Initializing prediction array with 0's
        g_z = np.zeros(m)
        for i in range(m):
            # Model
            f_wb = np.dot(X[i], w) + b
            g_z[i] = Logistic_Regression.sigmoid(f_wb)
            print(f"Predicted Value = {g_z[i]}, Actual Value = {Y[i]}")
        # Return predictions
        return g_z

    # Return function for parameters w and b
    def show_parameters(self):
        print(f"w = {self.w_final}, b = {self.b_final}")
        return self.w_final, self.b_final

    # Return function for cost history
    def show_cost_history(self):
        return self.cost
