import pandas as pd
from math import sqrt

df = pd.read_csv("../datasets/kc_house_data.csv")
print (df)

df_theta = [70,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
theta = pd.DataFrame(df_theta)
df_predicted_values = []
predicted_values = pd.DataFrame(df_predicted_values)
df_train = pd.read_csv("../datasets/kc_house_data.csv")
y = df['price']

train_size = 0.7
train_end = int(len(df)*train_size)

y_train = y[:train_end]
y_test = y[train_end:]
df.drop(columns=['id', 'date', 'price'], axis=1, inplace=True)

y_test = pd.DataFrame(y_test)


df_train = df[:train_end]
df_test = df[train_end:]

df_train = pd.DataFrame(df_train)
print("df train data",df_train.iloc[1,0])

learning_rate = 0.01
n_shape = df.shape[0]
train_n = y_train.shape[0]
test_n = y_test.shape[0]
z = theta.shape[0]
training_shape = df_train.shape[0]

print("training shape", training_shape)
def MSE_of_houses_train(n):
        return print("error on training set: ", 1/n(y_train*predicted_values[:train_end]), "%")

def MSE_of_houses_test(n):
        return print("error on test set: ", 1/test_n(y_test*predicted_values[train_end:]), "%")

def MSE_overall(n):
        print("Overall MSE: ", 1/n(y_test*predicted_values), "%")

def x_i(i,j):
        df_train.iloc[i,j]
    

def y_i(i):
        y_test.iloc[price,0]

def h_of_x(n):
        (theta.iloc[0]*1  + theta[0]*df_train.iloc[n,0] + theta.iloc[1]*df_train.iloc[n,1] + theta.iloc[2]*df_train.iloc[n,2] + theta.iloc[3]*df_train.iloc[n,3] + theta.iloc[4]*df_train.iloc[n,4] + theta.iloc[5]*df_train.iloc[n,5]
        + theta.iloc[6]*df_train.iloc[n,6] +theta.iloc[7]*df_train.iloc[n,7] +theta.iloc[8]*df_train.iloc[n,8] +theta.iloc[9]*df_train.iloc[n,9] + theta.iloc[10]*df_train.iloc[n,10] + theta.iloc[11]*df_train.iloc[n,11] +
        theta.iloc[12]*df_train.iloc[n,12] + theta.iloc[13]*df_train.iloc[n,13] + theta.iloc[14]*df_train.iloc[n,14] + theta.iloc[15]*df_train.iloc[n,15] + theta.iloc[16]*df_train.iloc[n,16] +theta.iloc[17]*df_train.iloc[n,17])

def j_of_theta(iterations):
    for p in range(0,iterations):
        for theta_j in range(17):
                for n in range(0, n_shape):
                        print("z:", theta_j)
                        print("theta:", theta[0])
                        theta[theta_j] = theta[theta_j] * learning_rate * sqrt(h_of_x(n) - y_i(n))*x_i(n,theta_j)


def predict(n):
    (theta[0]*1 + theta[0]*df_train[n,0] + theta[0]*df_train[n,1] + theta[1]*df_train[n,2] + theta[2]*df_train[n,3] + theta[3]*df_train[n,4] + theta[4]*df_train[n,5] + theta[5]*df_train[n,6] +theta[6]*df_train[n,7] +
     theta[7]*df_train[n,8] +theta[8]*df_train[n,9] + theta[9]*df_train[n,10] + theta[10]*df_train[n,11] + theta[11]*df_train[n,12] + theta[12]*df_train[n,13] + theta[13]*df_train[n,14] + theta[14]*df_train[n,15] +
     theta[15]*df_train[n,16] +theta[16]*df_train[n,17] + theta[17]*df_train[n,18] + theta[18]*df_train[n,19] + theta[19]*df_train[n,20])

def main():
    j_of_theta(1)
    for p in range(0, train_end):
        predicted_values.append(predict(p))

    MSE_of_houses_train(train_n)
    MSE_of_houses_test(test_n)
    MSE_overall(n_shape)
""" run through h_of_x with the last 30% of the dataset to make predictions and plot also look up how to calculate the the accuracy of prediction from scratch jason brownlee does show how to do this in
one of his books """



if __name__ == "__main__":
    main()



