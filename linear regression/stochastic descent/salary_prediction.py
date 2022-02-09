import pandas as pd
from math import sqrt
import numpy as np
import statistics
import matplotlib.pyplot as plt
df = pd.read_csv("../datasets/Salary.csv")

print (df)
df_ts = pd.DataFrame(df['YearsExperience'])
y = df['Salary']
df_ts['b'] = 1
print(df_ts)
weightsmps = ((df_ts.T.dot(df_ts))**-1).dot(df_ts.T.dot(y))
print("MOORE PENROSE WEIGHTS",weightsmps)

train_size = 0.7
train_end = int(len(df)*train_size)

df_train = df[:train_end]
df_test = df[train_end:]

y_train = y[:train_end]
y_test = y[train_end:]

y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)

train_n = df_train.shape[0]
print("train n shape", train_n)
theta = [31000, 8000]
theta = pd.DataFrame(theta)
def MSE_of_houses_train(n):
        return print("error on training set: ", 1/n(y_train*train_pred)**2, "%")

def MSE_of_houses_test(n):
        val = 0
        for i in range(n):
                """"""
                val = val + (y_test.iloc[n-1] * predicted_values.iloc[n-1])**2
        return (print ("error: ", 1/test_n*val))

def MSE_overall(n):
        return print("Overall MSE: ", 1/n(y_test*predicted_values), "%")

def x_i(i,j):
        return(df_train.iloc[i,j])
    

def y_i(i):
        return(float(y.iloc[i]))

def h_of_x(n):
        df_p=pd.DataFrame({'b' : 1, 'YearsExperience': df_train.iloc[n,0]}, index=[0])
        return(np.dot(theta.values.flatten(), df_p.values.flatten()))

def j_of_theta(iterations):
         for n in range(0, train_n):
                 if iterations < 1:
                         break
                 print("Iterations left: ", iterations)
                 iterations = iterations - 1
                 for theta_j in range(0,2):
                         print("theta:", theta.iloc[theta_j])
                         print("WEIGHTS:", theta)
                         if theta_j == 0:
                                 theta.loc[theta_j] = theta.iloc[theta_j] + .0000006828  * 1/train_n * (y_i(n) - h_of_x(n))
                                 print("CHANGED WEIGHT", theta.iloc[theta_j])
                         else:
                                 theta.loc[theta_j] = theta.iloc[theta_j] + .0000006828 * 1/train_n *(y_i(n) - h_of_x(n))*x_i(n,theta_j)
                     


def predict(n):
    return(round(theta.iloc[0]*1 + theta.iloc[1]*df_ts.iloc[n,0]))





def main():
    print("Count:", y.shape)
    print("Y**2=", y.sum()**2)
    print("SUM OF YOE: ", np.sum(df_ts["YearsExperience"]))
    print("SUM OF SALARYS: ", y.sum())
    ytx = y.dot(df_ts["YearsExperience"])
    print("YTX: ", ytx)
    print("X**2=", df_ts["YearsExperience"].sum()**2)
    print("X * Y = ", ytx)
    j_of_theta(150000)
    print(theta)
    for prediction in range(0,34):
            print("PREDICTION:", predict(prediction))
            print("Actual:", y_i(prediction))
    """print("Slope:", getSlope((x1, y1), (x2, y1)))
    the_mean = sum(y_train)/len(y_train)
    print("MEAN OF Y:", the_mean)
    j_of_theta(5000)
    for ft in range (0,50):
            print("prediction:",h_of_x(ft), "vs actual: ",  y_i(ft))"""
    """for n in range(0, train_n):
                                 train_pred = []
                                 train_pred_value = h_of_x(n)
                                 print(train_pred_value)
                                 train_pred.append(train_pred_value)
                                 train_pred = pd.DataFrame(train_pred)"""
    """MSE_of_houses_train(train_n)
    print("train end:", train_end,"test n shape:", test_n)
    for p in range(0, test_n):
            print("DF TEST: ", df_test)
            print(df_test.iloc[2,1])
            pv = predict(1)
            print("Predicted value:", pv)
            predicted_values.append(pv, ignore_index=True)"""
            
    """print("Shape of predicted values", predicted_values.shape[0])
    print("Shape of test shape:", df_test.shape[0])
    MSE_of_houses_test(test_n)
    for p in range(0, train_end):
        predicted_values.append(predict(p))

    MSE_of_houses_train(train_n)
    MSE_of_houses_test(test_n)
    MSE_overall(n_shape)"""




if __name__ == "__main__":
    main()



