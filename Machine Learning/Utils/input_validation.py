import numpy as np

# Validating inputs
def check_inputs(x_train, y_train, learning_rate, number_of_iterations, dimesion):
    if x_train.ndim != dimesion:
        print("Its regression with single variable not multi.")
        return False
    if not np.any(x_train):
        print("Invalid training.")
        return False
    if not np.any(y_train):
        print("Invalid testing set.")
        return False
    if learning_rate == 0:
        print("Learning rate should be greater than 0.")
        return False
    if learning_rate > 1:
        print("Learning rate should be less than 1.")
        return False
    if number_of_iterations == 0:
        print("Number of iterations should be greater than 0.")
        return False
    return True