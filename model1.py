import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import math
from sklearn.metrics import accuracy_score, f1_score, auc, roc_auc_score, roc_curve
import matplotlib.pyplot as plt


def read_file(file_path):  # read data into the algo
    data_df = pd.read_csv(file_path, header=None)
    # no heads here, the program will take the first row as keys by default
    ele_keys = list(range(0, 57, 1))
    x = data_df[ele_keys].values
    y = np.array(data_df[57].values)
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    return x, y


X, Y = read_file("spambase.data")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


def sigmoid(X):
    return 1 / (1 + math.exp(-X))


def log_loss(y, y_):
    return -1 * (y * math.log(sigmoid(y_)) + (1 - y) * math.log(1 - sigmoid(y_)))


def cal_grad(y, y_, x):
    return (y - sigmoid(y_)) * (x.reshape(-1, 1))


def adam(X, Y, epochs=100):
    # These values are taken as per the research paper
    alpha = 0.01
    beta_1 = 0.9
    beta_2 = 0.99

    M = X.shape[0]
    N = X.shape[1]
    w = np.random.uniform(-1, 1, size=N).reshape(N, 1)
    epsilon = 1e-8

    m_t = 0
    v_t = 0
    iter = 0
    cnt = 0
    for epoch in range(1, epochs + 1):
        cnt += 1
        for i in range(M):
            iter += 1
            y = Y[i]
            y_ = np.dot(w.T, X[i].reshape(N, 1))
            g_t = -(1.0 / M) * cal_grad(y, y_, X[i])  # change on the w

            m_t = beta_1 * m_t + (1 - beta_1) * g_t
            v_t = beta_2 * v_t + (1 - beta_2) * (g_t * g_t)
            m_hat = m_t / (1 - (beta_1 ** iter))
            v_hat = v_t / (1 - (beta_2 ** iter))
            w_prev = w  # iteration

            w = w - (alpha * m_hat) / (np.sqrt(v_hat) + epsilon)

        if epoch % 100 == 0:
            print(log_loss(y, y_))
        if (log_loss(y, y_) < 1e-10):
            break

    print("Converged at Epoch Number {} and total iteration done = {}".format(cnt, iter))

    return w


def sgd(X, Y, lr=0.1, epochs=200):
    M = X.shape[0]
    N = X.shape[1]
    w = np.zeros(N).reshape(N, 1)
    # b = np.random.uniform(-1, 1, size = 1)
    for epoch in range(1, epochs + 1):
        for i in range(M):
            y_ = np.dot(w.T, X[i].reshape(N, 1))
            y = Y[i]
            # dL/dw = -(y_i - sigmoid(x_i)) * x_i
            w_grad = -(1.0 / M) * (Y[i] - sigmoid(y_)) * (X[i].reshape(N, 1))
            w = w - lr * w_grad
        if epoch % 100 == 0:
            print("Loss : {}".format(log_loss(y, y_)))
    return w


def predict(X, w):
    y_pred = []
    for i in range(len(X)):
        y_pred.append(np.round(sigmoid(np.dot(w.T, X[i].reshape(X.shape[1], 1)))))
    return y_pred


def adam_train(x_input, y_input, x_veri, y_veri):
    mean_error = []
    std_error = []
    train_percent_list = [10, 20, 30]
    for percent in train_percent_list:
        spec_error_list = []
        for i in range(10):
            number_used = int(x_input.shape[0] * percent / 100)
            x_actual_used = x_input[:number_used]
            y_actual_used = y_input[:number_used]
            w = adam(x_actual_used, y_actual_used)
            y_pred = predict(x_veri, w)
            error_rate = 1 - accuracy_score(y_veri, y_pred)
            spec_error_list.append(error_rate)
        percent_error_mean = np.mean(spec_error_list)
        percent_error_std = np.std(spec_error_list)
        mean_error.append(percent_error_mean)
        std_error.append(percent_error_std)
    return mean_error, std_error


def sgd_train(x_input, y_input, x_veri, y_veri):
    mean_error = []
    std_error = []
    train_percent_list = [10, 20, 30]
    for percent in train_percent_list:
        spec_error_list = []
        for i in range(10):
            number_used = int(x_input.shape[0] * percent / 100)
            x_actual_used = x_input[:number_used]
            y_actual_used = y_input[:number_used]
            w = sgd(x_actual_used, y_actual_used)
            y_pred = predict(x_veri, w)
            error_rate = 1 - accuracy_score(y_veri, y_pred)
            spec_error_list.append(error_rate)
        percent_error_mean = np.mean(spec_error_list)
        percent_error_std = np.std(spec_error_list)
        mean_error.append(percent_error_mean)
        std_error.append(percent_error_std)
    return mean_error, std_error


def graph_plot(mean_sgd, std_sgd, mean_adam, std_adam):
    colors = ['red', 'blue', 'green']
    x1 = np.arange(len(mean_sgd))
    x2 = np.arange(len(mean_adam))

    plt.plot(x1, mean_sgd, marker='o', linestyle='-', color='cyan',
             label='sgd')
    for i, (mean_i, std_i, color) in enumerate(zip(mean_sgd, std_sgd, colors)):
        plt.fill_between([x1[i], x1[i]], mean_i - std_i,
                         mean_i + std_i, color='black', alpha=1,
                         label='sgd Std Dev' if i == 0 else "")

    plt.plot(x2, mean_adam, marker='o', linestyle='-', color='red',
             label='adam')
    for i, (mean_i, std_i, color) in enumerate(zip(mean_adam, std_adam, colors)):
        plt.fill_between([x2[i], x2[i]], mean_i - std_i,
                         mean_i + std_i, color='grey', alpha=1,
                         label='adam Std Dev' if i == 0 else "")

    plt.xticks(x1, ['10%', '20%', '30%'])
    plt.xlabel('Percentage of data')
    plt.ylabel('Error Rate')
    # plt.title('MODEL2: one-hidden layer with 10 neurons and relu')
    plt.legend()
    plt.show()


adam_mean, adam_std = adam_train(X_train, Y_train, X_test, Y_test)
sgd_mean, sgd_std = sgd_train(X_train, Y_train, X_test, Y_test)
graph_plot(sgd_mean, sgd_std, adam_mean, adam_std)
