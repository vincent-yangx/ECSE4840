import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import torch

def read_file(file_path):  # read data into the algo
    data_df = pd.read_csv(file_path, header=None)
    # no heads here, the program will take the first row as keys by default
    ele_keys = list(range(0, 57, 1))
    x = data_df[ele_keys].values
    y = np.array(data_df[57].values)
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    return x, y


def build_model(input_shape):
    model = Sequential([
        # choose model here
        # 1 Hidden layer with 10 neurons and relu activation
        # Dense(10, activation='relu', input_shape=(input_shape,)),

        # 1 Hidden layer with 10 neurons and tahn activation

        # Dense(10, activation='tanh', input_shape=(input_shape,)),
        # 1 Hidden layer with 30 neurons and relu activation

        # Dense(30, activation='relu', input_shape=(input_shape,)),
        # 2 Hidden layer with 10 neurons relu activation in 1st layer and the same for 2nd layer
        Dense(10, activation='relu', input_shape=(input_shape,)),
        Dense(10, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model


def compile_adam(model, train_data, train_output, epochs=20, batch_size=20):
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(train_data, train_output, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    return history


def compile_sgd(model, train_data, train_output, epochs=20, batch_size=32):
    # Create an SGD optimizer instance with custom parameters
    sgd_optimizer = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=sgd_optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(train_data, train_output, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    return history


def test_model(model, x_test, y_test):
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    return test_loss, test_accuracy


train_percent_list = [10,20,30]

def train_for_sgd():
    mean_error_rates = []
    std_error_rates = []

    for train_percent in train_percent_list:
        error_rates = []

        for j in range(10):
            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

            actual_test_size = int((train_percent / 100) * X_train.shape[0])
            X_train_actual = X_train[:actual_test_size]
            y_train_actual = y_train[:actual_test_size]

            # Build, compile and train the model
            model = build_model(X_train_actual.shape[1])
            history = compile_sgd(model, X_train_actual, y_train_actual)

            # Evaluate the model
            test_loss, test_accuracy = test_model(model, X_test, y_test)
            error_rate = 1 - test_accuracy
            error_rates.append(error_rate)

        # Calculate mean and standard deviation for the current percentage
        mean_error_rate = np.mean(error_rates)
        std_error_rate = np.std(error_rates)

        mean_error_rates.append(mean_error_rate)
        std_error_rates.append(std_error_rate)
    return mean_error_rates, std_error_rates


def train_for_adam():
    mean_error_rates = []
    std_error_rates = []

    for train_percent in train_percent_list:
        error_rates = []

        for _ in range(10):
            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=None)

            # Select a subset of the training data based on the current percentage
            subset_size = int((train_percent / 100) * X_train.shape[0])
            X_train_actual = X_train[:subset_size]
            y_train_actual = y_train[:subset_size]

            # Build, compile and train the model
            model = build_model(X_train_actual.shape[1])
            history = compile_adam(model, X_train_actual, y_train_actual)

            # Evaluate the model
            test_loss, test_accuracy = test_model(model, X_test, y_test)
            error_rate = 1 - test_accuracy
            error_rates.append(error_rate)

        # Calculate mean and standard deviation for the current percentage
        mean_error_rate = np.mean(error_rates)
        std_error_rate = np.std(error_rates)

        mean_error_rates.append(mean_error_rate)
        std_error_rates.append(std_error_rate)
    return mean_error_rates, std_error_rates

file_path = 'spambase.data'
x, y = read_file(file_path)
mean_sgd, std_sgd = train_for_sgd()
mean_adam, std_adam = train_for_adam()



def graph_plot(mean_sgd, std_sgd,mean_adam,std_adam):
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
    plt.legend()
    plt.show()


graph_plot(mean_sgd, std_sgd,mean_adam,std_adam)

