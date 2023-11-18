import numpy as np
import pandas as pd
from numpy import linalg as LA
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def read_file(file_path):  # read data into the algo
    data_df = pd.read_csv(file_path, header=None)
    row_count = len(data_df.index)
    col_count = data_df.columns[-1]
    ele_keys = list(range(1, data_df.columns[-1] + 1, 1))
    x = data_df[ele_keys].values
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    y = np.array(data_df[0].values)
    y = y - 2
    return row_count, col_count, x, y


def check_stop_condition(para_1, para_2):  # stop condition
    if pow(LA.norm(para_1 - para_2), 2) <= 0.001:
        return True
    else:
        return False


def train_set_choice(_num, x, y):  # select train set
    if _num == 1:
        random_sample_row = np.random.choice(x.shape[0], size=1)
        x_actual_train = x[random_sample_row]
        y_actual_train = y[random_sample_row]
    else:
        x_n = x[y == -1]
        x_p = x[y == 1]
        y_n = y[y == -1]
        y_p = y[y == 1]
        num_n = int(x_n.shape[0] * _num / x.shape[0])
        num_p = _num - num_n
        n_rdm_sample_row = np.random.choice(x_n.shape[0], size=num_n)
        p_rdm_sample_row = np.random.choice(x_p.shape[0], size=num_p)
        n_train = x_n[n_rdm_sample_row]
        n_result = y_n[n_rdm_sample_row]

        p_train = x_p[p_rdm_sample_row]
        p_result = y_p[p_rdm_sample_row]

        x_actual_train = np.vstack((n_train, p_train))
        y_actual_train = np.hstack((n_result, p_result))
    return x_actual_train, y_actual_train


def mysgdsvm(filename, k, numruns):  # implement the algorithm
    rows, cols, x_csv, y_csv = read_file(filename)
    para_lambda = 1
    loss_list = []
    time_list = []
    for i in range(0, numruns, 1):
        w = np.zeros(x_csv.shape[1])  # initialize w
        # w = np.ones(cols) * 0.01
        start_time = time.time()    # record time
        spec_loss_list = []
        iter = 1
        while True:
            train_set, train_label = train_set_choice(k, x_csv, y_csv)
            sum = np.zeros(x_csv.shape[1])
            loss_total = 0
            # print(w)
            for j in range(train_set.shape[0]):     # iterate to find the false classification
                if train_label[j] * (w.dot(train_set[j])) < 1:
                    sum += train_label[j] * train_set[j]    # vector sum
                    loss_total += 1 - train_label[j] * (w.dot(train_set[j]))    # objective sum

            lr_rate = 1 / (para_lambda * iter)
            w_half = (1 - lr_rate * para_lambda) * w + lr_rate * sum / k
            w_next = min(1, 1 / ((LA.norm(w_half))+pow(10,-15))) * w_half
            loss_here = 0.5 * para_lambda * pow(LA.norm(w), 2) + loss_total / k
            spec_loss_list.append(loss_here)
            flag = check_stop_condition(w, w_next)
            if flag:    # check stop condition
                w = w_next
                loss_list.append(spec_loss_list)
                duration = time.time() - start_time
                time_list.append(duration)
                break
            else:
                iter += 1
                w = w_next
    horiz = []      # code for plot
    for z in range(0, numruns, 1):
        aux = [k for k in range(0, len(loss_list[z]), 1)]
        horiz.append(aux)

    for l in range(0, numruns, 1):
        str = "run{}".format(l)
        plt.plot(horiz[l], loss_list[l], label=str)

    plt.legend()
    plt.title('k={}'.format(k))
    plt.xlabel('iterations')
    plt.ylabel('prime objective function')
    plt.show()

    return loss_list, time_list

# we choose k among 1,20,100,200,2000
#loss is the vector storing the loss we get when we use the same k for 5 times
#time is the vector storing the running time of each run
loss, time = mysgdsvm("MNIST-13.csv", 1, 5)
print("this is overall time:{}\nthis is the average run time:{}\nthis is the"
      " time std:{}\n".format(time,np.mean(time),np.std(time)))


