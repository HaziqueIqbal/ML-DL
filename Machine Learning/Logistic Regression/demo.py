from Logist_Regression import Logistic_Regression
import numpy as np



def main():

    total_iterations = 100000
    learning_rate = 0.1
    lambda_ = 0

    # ------------------------------------------TEST--------------------------------------------------

    # Multi variable
    X_train = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [
                       3, 0.5], [2, 2], [1, 2.5]])
    Y_train = np.array([0, 0, 0, 1, 1, 1])

    lrg = Logistic_Regression(X_train, Y_train, learning_rate,
                              total_iterations, lambda_)
    w, b = lrg.show_parameters()
    lrg.predict(X_train, Y_train, w, b)

    # ------------------------------------------TEST-END----------------------------------------------


if __name__ == "__main__":
    main()
