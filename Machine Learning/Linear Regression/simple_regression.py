import numpy as np
import math
import copy
from Utils.input_validation import check_inputs

# Linear Regression Model
class Simple_Linear_Regression:
    # Init function
    def __init__(self, x_train, y_train, learning_rate, number_of_iterations, lambda_=1):
        # Checking for valid inputs
        if not (check_inputs(x_train, y_train, learning_rate, number_of_iterations, 1)):
            exit()
        # Extracting shape of training set
        m = x_train.shape[0]
        # Initializing parameter with 0's
        w_init = 0.
        b_init = 0.
        # Getting values from Linear Regression Model
        self.cost, self.parameters, self.w_final, self.b_final = Simple_Linear_Regression.gradient_descent(
            x_train, y_train, w_init, b_init, learning_rate, number_of_iterations, lambda_)

    # Cost function
    def cost_function(X, Y, w, b, lambda_=1):
        # Extracting shape of training set
        m = X.shape[0]
        # Initializing costs to 0
        cost = 0
        reg_cost = 0
        for i in range(m):
            # Model
            f_wb = np.dot(X[i], w) + b
            # Squared error
            error = (f_wb - Y[i])**2
            # Sum of errors
            cost += error
        # Regularization cost
        reg_cost = (w**2)
        # Multiplying with lambda
        reg_cost *= lambda_
        # Adding two costs and dividing it by total training examples
        total_cost = (cost + reg_cost) / (2 * m)
        # Return total cost
        return total_cost

    # Calulcating the derivate of the cost function w.r.t "w" and "b" parameters
    def gradient_function(X, Y, w, b, lambda_=1):
        # Extracting shape of training set
        m = X.shape[0]
        # Initializing derivative variables with 0's
        dj_dw = 0.
        dj_db = 0.

        for i in range(m):
            # Model
            f_wb = np.dot(X[i], w) + b
            # error
            error = f_wb - Y[i]
            # Multiplying error with feature
            dj_dw += error * X[i]
            dj_db += error
        # Dividing derivative variable with total training examples
        dj_dw /= m
        dj_db /= m
        # Regularization of paramter w
        dj_dw += (lambda_ * w) / m
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
            dj_dw, dj_db = Simple_Linear_Regression.gradient_function(
                X, Y, w, b, lambda_)

            # Chnage in variable w and b
            w = w - learning_rate * dj_dw
            b = b - learning_rate * dj_db

            # Appending change in history variables
            cost_history.append(
                Simple_Linear_Regression.cost_function(X, Y, w, b, lambda_))
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
        f_wb = np.zeros(m)
        
        for i in range(m):
            # Model
            f_wb[i] = np.dot(X[i], w) + b
            print(f"Predicted Value = {f_wb[i]}, Actual Value = {Y[i]}")
        # Return predictions
        return f_wb

    # Return function for parameters w and b
    def show_parameters(self):
        print(f"w = {self.w_final}, b = {self.b_final}")
        return self.w_final, self.b_final

    # Return function for cost history
    def show_cost_history(self):
        return self.cost


