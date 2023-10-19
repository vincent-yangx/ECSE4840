import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import random
from numpy import linalg as LA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import json
import ast


# data preprocessing
def read_file(file_path):  # read data into the algo
    data_df = pd.read_csv(file_path, header=None)
    # no heads here, the program will take the first row as keys by default
    ele_keys = list(range(0, 57, 1))
    x = data_df[ele_keys].values
    y = np.array(data_df[57].values)
    x_ns = x[y == 0]
    x_s = x[y == 1]
    y_ns = y[y == 0]
    y_s = y[y == 1]
    return x_ns, x_s, y_ns, y_s


def split_dataset(x_ns, x_s, y_ns, y_s, training_size=0.8):  # split into train and test
    size_test = 1 - training_size
    x_ns_train, x_ns_test, y_ns_train, y_ns_test = train_test_split(x_ns, y_ns, test_size=size_test)
    x_s_train, x_s_test, y_s_train, y_s_test = train_test_split(x_s, y_s, test_size=size_test)

    x_train = np.vstack((x_ns_train, x_s_train))
    y_train = np.hstack((y_ns_train, y_s_train))

    x_test = np.vstack((x_ns_test, x_s_test))
    y_test = np.hstack((y_ns_test, y_s_test))

    return x_train, y_train, x_test, y_test


def actual_train_set(x_train, y_train, train_percent):  # get the actual train_set
    actual_train_size = int(x_train.shape[0] * train_percent / 100)
    random_sample_rows = np.random.choice(x_train.shape[0], size=actual_train_size)
    x_actual_train = x_train[random_sample_rows]
    y_actual_train = y_train[random_sample_rows]
    return x_actual_train, y_actual_train


# logistic regression part (use sigmoid function here)
def sigmoid(x, theta):
    # print(x.shape, theta.shape)
    return 1 / (1 + np.exp(-x.dot(theta)))


def loss_logistic(x, y, theta, lr=0.01):  # loss function
    loss = -y * np.log(sigmoid(x, theta)) - (1 - y) * np.log(1 - sigmoid(x, theta))
    total_loss = np.mean(loss, axis=0)
    final_loss = total_loss / x.shape[0] + lr * pow(LA.norm(theta), 2) / 2
    return final_loss


def gradient(x, y, theta, lr=0.01): # the gradient function
    dev = y - sigmoid(x, theta)
    grad = -x.T.dot(dev)
    final_grad = grad / x.shape[0] + lr * theta
    return final_grad


def logistics_train(x, y, alpha=1E-3, lr=0.01, max_iter=100):   # train the model
    cols = x.shape[1]
    theta = np.zeros(cols)
    for i in range(max_iter):
        grad = gradient(x, y, theta, lr)
        theta = theta - alpha * grad
    return theta


def logistics_test(x, y, theta):        # get the result from the test_set
    probabilities = sigmoid(x, theta)
    predictions = (probabilities >= 0.5).astype(int)
    error = np.sum(predictions != y)
    error_rate = error / (2 * x.shape[0])
    return error_rate


def logisticsRegression(filename, num_splits, train_percent):
    x_ns, x_s, y_ns, y_s = read_file(filename)
    error_record = np.zeros((num_splits, len(train_percent)))
    scaler = StandardScaler()
    # Fit the scaler on the entire dataset and transform it
    x_ns = scaler.fit_transform(x_ns)
    x_s = scaler.transform(x_s)
    for i in range(num_splits):
        x_train, y_train, x_test, y_test = split_dataset(x_ns, x_s, y_ns, y_s)
        for j in range(len(train_percent)):
            percent = train_percent[j]
            x_actual_train, y_actual_train = actual_train_set(x_train, y_train, percent)
            theta_result = logistics_train(x_actual_train, y_actual_train)
            error_result = logistics_test(x_test, y_test, theta_result)
            error_record[i][j] = error_result
    logistics_mean = np.mean(error_record, axis=0)
    logistics_std = np.std(error_record, axis=0)
    print("These are the mean and std list for different percents in Logistics Regression")
    print(logistics_mean)
    print(logistics_std)
    return logistics_mean, logistics_std


# train_percent_list = [5, 10, 15, 20, 25, 30]
# mean1,std1=logisticsRegression("spambase.data", 10, train_percent_list)


# naive Bayes Gaussian

def feature_compute(x, y):
    # get the std, mean for each feature, and the pre-conditional probabilities
    x_ns = x[y == 0]
    x_s = x[y == 1]
    mean_ns = np.mean(x_ns, axis=0)
    mean_s = np.mean(x_s, axis=0)
    std_ns = np.std(x_ns, axis=0)
    std_s = np.std(x_s, axis=0)
    p_s = np.mean(y)
    p_ns = 1 - p_s
    return mean_ns, mean_s, std_ns, std_s, p_ns, p_s


def con_prob(x, mean, std, pro):    # conditional probability
    std += 1E-12
    cond_p = 1 / (np.sqrt(2 * np.pi) * std) * np.exp(-1 / 2 * (x - mean) ** 2 / (std ** 2))
    total_cond_p = np.prod(cond_p, axis=1) * pro
    return total_cond_p


def predict(x, y, mean_ns, mean_s, std_ns, std_s, p_ns, p_s):   # predict the result
    prob_y0 = con_prob(x, mean_ns, std_ns, p_ns)
    prob_y1 = con_prob(x, mean_s, std_s, p_s)
    return prob_y0, prob_y1


def error_cal(p0, p1, y):   # calculate the error rate
    predict = np.where(p1 > p0, 1, 0)
    errors = np.sum(predict != y)
    error_rate = errors / len(y)
    return error_rate


def naiveBayesGaussian(filename, num_splits, train_percent):
    x_ns, x_s, y_ns, y_s = read_file(filename)
    error_record = np.zeros((num_splits, len(train_percent)))
    for i in range(num_splits):
        x_train, y_train, x_test, y_test = split_dataset(x_ns, x_s, y_ns, y_s)
        for j in range(len(train_percent)):
            percent = train_percent[j]
            x_actual_train, y_actual_train = actual_train_set(x_train, y_train, percent)
            mean_ns, mean_s, std_ns, std_s, p_ns, p_s = feature_compute(x_actual_train, y_actual_train)
            prob_y0, prob_y1 = predict(x_test, y_test, mean_ns, mean_s, std_ns, std_s, p_ns, p_s)
            error_rate = error_cal(prob_y0, prob_y1, y_test)
            error_record[i][j] = error_rate
    naive_mean = np.mean(error_record, axis=0)
    naive_std = np.std(error_record, axis=0)
    print("These are the mean and std list for different percent in Naive Bayes")
    print(naive_mean)
    print(naive_std)
    return naive_mean, naive_std


filename = input("Please input the filename:\n")    # prompt input and type conversion
num_splits = input("Please input the num_splits:\n")
num_splits = json.loads(num_splits)
train_percent_input = input("Please input the train_percent(format:[1,2,3]):\n")
train_percent = json.loads(train_percent_input)
mean_l, std_l = logisticsRegression(filename, num_splits, train_percent)
mean_n, std_n = naiveBayesGaussian(filename, num_splits, train_percent)


# Put two graphs together
colors = ['red', 'blue', 'green', 'cyan', 'black', 'yellow']
x1 = np.arange(len(mean_l))
x2 = np.arange(len(mean_l))

plt.plot(x1, mean_l, marker='o', linestyle='-', color='blue',
         label='mean and dev for Logistic Regression')
for i, (mean_i, std_i, color) in enumerate(zip(mean_l, std_l, colors)):
    plt.fill_between([x1[i], x1[i]], mean_i - std_i,
                     mean_i + std_i, color='blue', alpha=1)

plt.plot(x2, mean_n, marker='o', linestyle='-', color='red',
         label='mean and dev for Naive Bayes')
for i, (mean_i, std_i, color) in enumerate(zip(mean_n, std_n, colors)):
    plt.fill_between([x2[i], x2[i]], mean_i - std_i,
                     mean_i + std_i, color='red', alpha=1)

plt.xticks(x1, ['5%', '10%', '15%', '20%', '25%', '30%'])
plt.ylabel('Error')
plt.title('Logistics & naive bayes')
plt.legend()
plt.show()
