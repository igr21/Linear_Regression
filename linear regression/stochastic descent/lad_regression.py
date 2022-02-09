from numpy import power, array, diag, dot, maximum, empty, repeat, ones, sum, subtract, asmatrix, asarray, reshape, median
from numpy.linalg import pinv, inv
import pandas as pd
from math import sqrt
import statistics
import matplotlib.pyplot as plt
from sklearn import preprocessing
import tensorflow as tf
import tensorflow.python.ops.numpy_ops.np_config as np_config
import numpy as np


def RMSE_overall(error, pv):
    return (print("Overall RMSE: ", sqrt(error / pv)))

def predict(n, weights, df_test):
    weights=pd.DataFrame(weights)
    
    return abs((weights.iloc[0, 0]*df_test.iloc[n, 0] + weights.iloc[0, 1]*df_test.iloc[n, 1] + weights.iloc[0, 2]*df_test.iloc[n, 2] + weights.iloc[0, 3]*df_test.iloc[n, 3] + weights.iloc[0, 4]*df_test.iloc[n, 4] +
           weights.iloc[0, 5]*df_test.iloc[n, 5]+weights.iloc[0, 6]*df_test.iloc[n, 6] + weights.iloc[0, 7]*df_test.iloc[n, 7] + weights.iloc[0, 8]*df_test.iloc[n, 8] + weights.iloc[0, 9]*df_test.iloc[n, 9] +
           weights.iloc[0, 10]*df_test.iloc[n, 10] + weights.iloc[0, 11] *
           df_test.iloc[n, 11] + weights.iloc[0, 12]*df_test.iloc[n, 12]
           + weights.iloc[0, 13]*df_test.iloc[n, 13]))

def IRLS(y, X, maxiter, w_init=1, d=.00000001, tolerance=.000001):
    n, p = X.shape
    delta = array(repeat(d, n)).reshape(1, n)
    w = repeat(1,n)
    W = diag( w )
    x = pinv(X).dot(y)
    
    with tf.device('/gpu:0'):
        XT_W_X = inv(X.T.dot(W).dot(X))
        XT_W_y = X.T.dot(W).dot(y)
        B =  dot(XT_W_X, XT_W_y)
    print(B)
    
    for _ in range(maxiter):
        _B = B
        y = pd.DataFrame(y)
        _w = abs(X.dot(x) - y)

        w = pow(float(1)/maximum(delta, _w), -.5)
        rows = w.shape[0]
        b = sum(w[0])
        
        for 1 in w.shape:
            c = sum(w[i])
            if c > b:
                b = c
        W = diag( w[0] )

        with tf.device('/gpu:0'):
            XT_W_X = inv(X.T.dot(W).dot(X))
            XT_W_y = X.T.dot(W).dot(y)
            B =  dot(XT_W_X, XT_W_y)
            
            """B = dot(inv(X.T.dot(W).dot(X).dot(W)), (X.T.dot(W.T).dot(W).dot(y)))"""
        tol = sum(abs(B - _B))
        print("iteration:", _)
        print("Tolerance = %s" % tol)
        if tol <= tolerance:
            print("Tolerance = %s" % tol)
            return B

    return B


# Test Example: Fit the following data under Least Absolute Deviations regression
# first line = "p n" where p is the number of predictors and n number of observations
# following lines are the data lines for predictor x and response variable y
#	 "<pred_1> ... <pred_p> y"
# next line win "n" gives the number n of test cases to expect
# following lines are the test cases with predictors and expected response
def main():
    np_config.enable_numpy_behavior()
    df_predicted_values=[]
    error=0
    df = pd.read_csv("../datasets/kc_house_data.csv")
    print(df)
    df.insert(0, "b", 1)
    y = df['price']

    df.drop(columns=['id', 'date', 'price', 'lat', 'long',
            'zipcode', 'yr_renovated', 'floors'], axis=1, inplace=True)

    train_size = 0.7
    train_end = int(len(df)*train_size)
    y_test = y[train_end:]
    df_train = df[:train_end]
    y_train = y[:train_end]
    y = tf.convert_to_tensor(y_train, dtype=tf.float32)
    X = tf.convert_to_tensor(df_train, dtype=tf.float32)
    B = IRLS(y=y, X=df_train, maxiter=20)
    B.shape = (1,14)
    """"print(B)"""
    df_t = df[train_end:]
    for n in range(0, df_t.shape[0]):
        df_predicted_values.append(predict(n, B, df_t))
    df_predicted_val=asarray(df_predicted_values)
    df_test_np=y_test
    df_test_np=df_test_np.to_numpy()
    for n in range(0, df_t.shape[0]):
            """df_predicted_values = df_predicted_values.to_numpy()"""
            error=(error) + (df_predicted_val[n] - df_test_np[n])**2
            

    RMSE_overall(error, y_test.shape[0])

if __name__ == "__main__":
    main()
