# ML-DL
All about Machine and Deep Learning

#### *Install all requirments with ```pip install -r requirements.txt``` command.*

## Machine Learning
### Linear Regression with Regularization
#### 1) One Variable
<b>How to use?</b>

1. Import Linear Regression Module like, 
```from simple_regression import Simple_Linear_Regression```.
2. Set values for ```learning_rate```, ```lambda_```, ```number_of_iterations```.
3. Train your model, ```srg = Simple_Linear_Regression(X_train, Y_train, learning_rate, total_iterations, lambda_)```
4. Get your parameter values for <b>w</b> and <b>b</b>.
5. Test your model, ```srg.predict(X_train, Y_train, w, b)```
6. Demo available [here](https://github.com/HaziqueIqbal/ML-DL/blob/main/Machine%20Learning/Linear%20Regression/demo.py)

#### 2) Multi Variable
<b>How to use?</b>

1. Import Linear Regression Module like, 
```from multi_regression import Multiple_Linear_Regression```.
2. Set values for ```learning_rate```, ```lambda_```, ```number_of_iterations```.
3. Train your model, ```rgm = Multiple_Linear_Regression(X_train, Y_train, learning_rate, total_iterations, lambda_)```
4. Get your parameter values for <b>w</b> and <b>b</b>.
5. Test your model, ```rgm.predict(X_train, Y_train, w, b)```
6. Demo available [here](https://github.com/HaziqueIqbal/ML-DL/blob/main/Machine%20Learning/Linear%20Regression/demo.py)

### Logistic Regression with Regularization
<b>How to use?</b>

1. Import Logistic Regression Module like, 
```from Logist_Regression import Logistic_Regression``.
2. Set values for ```learning_rate```, ```lambda_```, ```number_of_iterations```.
3. Train your model, ```lrg = Logistic_Regression(X_train, Y_train, learning_rate, total_iterations, lambda_)```
4. Get your parameter values for <b>w</b> and <b>b</b>.
5. Test your model, ```lrg.predict(X_train, Y_train, w, b)```
6. Demo available [here](https://github.com/HaziqueIqbal/ML-DL/blob/main/Machine%20Learning/Logistic%20Regression/demo.py)

