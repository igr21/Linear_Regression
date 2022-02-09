from numpy import power, array, diag, dot, maximum, empty, repeat, ones, sum, subtract, asmatrix, asarray, reshape, median
from numpy.linalg import pinv, inv
import pandas as pd
from math import sqrt
import statistics
import matplotlib.pyplot as plt
from sklearn import preprocessing

df = pd.read_csv("../datasets/kc_house_data.csv")

y = df['price']
x1 = df['sqft_living15']
x2 = df['sqft_basement']

b = median(y)
print(x1)
print(df)
"""b,bedrooms,bathrooms,sqft_living,sqft_lot,waterfront,view,grade,condition, sqft_above,sqft_basement,yr_built,sqft_living15,sqft_lot15"""

df_theta = [1 for i in range(196)]

df.insert(0, "b", 1)
df_train = pd.read_csv("../datasets/kc_house_data.csv")


train_size = 0.7
train_end = int(len(df)*train_size)

y_train = y[:train_end]

y_test = y[train_end:]
df.drop(columns=['id', 'date', 'price', 'lat', 'long',
        'zipcode', 'yr_renovated', 'floors'], axis=1, inplace=True)

y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)


df_train = df[:train_end]

df_test = df[train_end:]
train_n = y_train.shape[0]

b_val = pd.DataFrame(df_train).to_numpy()

df_train = pd.DataFrame(df_train)
df_test = pd.DataFrame(df_test)

learning_rate = 0.01
n_shape = df.shape[0]

test_n = y_test.shape[0]

training_shape = df_train.shape[0]


theta = pd.DataFrame(df_theta)

theta = pd.DataFrame(theta)


z = theta.shape[0]
print("training shape", training_shape)


def RMSE_overall(error, pv):
    return (print("Overall RMSE: ", sqrt(error / pv)))


def x_i(i, j):
    return(df_train.iloc[i, j])


def y_i(i):
    return(y_train.iloc[i])


def h_of_x(n, W):
    df_row = df_train.iloc[n, ]
    df_row = pd.DataFrame(df_row)
    W = pd.DataFrame(W)
    return(round(np.dot(W.T.values.flatten(), df_row.values.flatten())))


def IRLS(y, A, maxiter, w_init=1, d=0.0001, tolerance=0.001,):
    n, p = A.shape
    fs = pow(14,2)
    w = repeat(1, fs)
    w.shape = (14, 14)
    print("w:", w[0], "w.shape", w.shape)
    
    x = pinv(A).dot(y)
    print("pinvxdoty:", x)
    """x.shape = (1,14)"""
    for _ in range(maxiter):
        A = asarray(A)
        x = asarray(x)
        x.shape = (1,14)
        e = abs(subtract(A.dot(x.T), y))
        w = pow(abs(e), 14-2/2)
        print("w", w.shape)
        w = asarray(w)
        w.reshape(15129,-14)
        print(w.shape)
        W=diag(w/sum(w))

        W = asmatrix(W)
        y = asmatrix(y)
        WA = W.dot(A)
        er
    return x


def j_of_theta(iterations):
    for n in range(0, train_n):
        if iterations < 1:
            break
        print("Iterations left: ", iterations)
        iterations=iterations - 1
        for theta_j in range(0, 14):
            print("z:", theta_j)
            print("real price:", y_i(n))
            print("predicted price", h_of_x(n))
            print("theta:", theta.iloc[theta_j])
            if theta_j == 0:
                theta.loc[theta_j]=theta.iloc[theta_j] - .000000000001 * (h_of_x(n) - y_i(n))
            else:
                theta.loc[theta_j]=theta.iloc[theta_j] - .000000000001 * (h_of_x(n) - y_i(n)) * x_i(n, theta_j)
            print("Should be updated theta:", theta.iloc[theta_j])
            print("WEIGHTS:", theta)
            print("TRAIN N:", train_n)


def cost_function(params, X, y):
    return np.sum(np.abs(y - fit(X, params)))


def simplex_alg():
    # the simplex algorithm goes through as many items as there is weights i.e 14 columns/dependent vars 14 weights therefore there will be 14 slack variables using the simplex algorithm go through the data in blocks of 14
    # and find the values of these slack variables then using the inequalties and equations find the LAD and slightly increase the weight if need be or minus then I am gonna guess its
    x=0
    return x


def e(lms, bhat):
    for n in range(0, train_n):
        bhat=bhat + np.absolute(y_i(n) - h_of_x(n, lms))
    return bhat


def predict(n, weights):
    weights=pd.DataFrame(weights)
    return(weights.iloc[0, 0]*df_test.iloc[n, 0] + weights.iloc[0, 1]*df_test.iloc[n, 1] + weights.iloc[0, 2]*df_test.iloc[n, 2] + weights.iloc[0, 3]*df_test.iloc[n, 3] + weights.iloc[0, 4]*df_test.iloc[n, 4] +
           weights.iloc[0, 5]*df_test.iloc[n, 5]+weights.iloc[0, 6]*df_test.iloc[n, 6] + weights.iloc[0, 7]*df_test.iloc[n, 7] + weights.iloc[0, 8]*df_test.iloc[n, 8] + weights.iloc[0, 9]*df_test.iloc[n, 9] +
           weights.iloc[0, 10]*df_test.iloc[n, 10] + weights.iloc[0, 11] *
           df_test.iloc[n, 11] + weights.iloc[0, 12]*df_test.iloc[n, 12]
           + weights.iloc[0, 13]*df_test.iloc[n, 13])


def main():
    print("median of y: ", statistics.median(y))
    error=0
    df_predicted_values=[]

    """j_of_theta(150)"""
    weights_LAD=IRLS(y_train, df_train, 10, w_init=1,
                       d=0.0001, tolerance=0.001)
    weights_LAD=diag(weights_LAD)
    weights_LAD.shape = (1,14)
    print(weights_LAD)
    for n in range(0, df_test.shape[0]):
        df_predicted_values.append(predict(n, weights_LAD))

    """df_predicted_values = pd.DataFrame(df_predicted_values)"""
    df_predicted_val=asarray(df_predicted_values)
    df_test_np=y_test
    df_test_np=df_test_np.to_numpy()
    for n in range(0, df_test.shape[0]):
        """df_predicted_values = df_predicted_values.to_numpy()"""
        error=(error) + (df_predicted_val[n] - df_test_np[n])**2
        print("predicted: ", df_predicted_val[n], " Actual: ", df_test_np[n])

    RMSE_overall(error, y_test.shape[0])


if __name__ == "__main__":
    main()
