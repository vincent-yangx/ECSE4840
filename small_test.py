"""
Author: Yixiang Wang
ECSE4840 Homework3 
"""
# Import all the libaries required
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

'''
Here we read the file and import data 
we split the features X and result y. 
'''


def filereading(filename):
    df = pd.read_csv(filename)  # read the data from the file
    data = np.array(df)  # convert the data into arrays.
    X = np.array(data[:, :-1])
    y = np.array(data[:, -1:])
    y = y.ravel()  # apply a transpose
    return X, y


'''
Here we process the data by grouping spam features
into X0 and non-spam features into X1.Also group spam label
into y1 and non-spam label into y0.  
'''


def dataproecss(X, y):
    # Classifying the non-spam features and spam features.
    indices_1 = np.where(y == 1)[0]  # get indices for where y=1
    indices_0 = np.where(y == 0)[0]  # get indices for where y=0

    X1 = X[indices_1]
    X0 = X[indices_0]
    y1 = y[indices_1]
    y0 = y[indices_0]
    return X0, X1, y0, y1


# This function is to split the data into 80:20
def datasplit(X0, X1, y0, y1):
    # Conduct randomization
    indices_2 = np.random.permutation(len(y0))
    X0s = X0[indices_2]
    y0s = y0[indices_2]
    indices_3 = np.random.permutation(len(y1))
    X1s = X1[indices_3]
    y1s = y1[indices_3]

    split_idx_0 = int(0.8 * len(X0))
    split_idx_1 = int(0.8 * len(X1))

    X_tr = np.vstack((X0s[:split_idx_0], X1s[:split_idx_1]))
    y_tr = np.hstack((y0s[:split_idx_0], y1s[:split_idx_1]))
    X_test = np.vstack((X0s[split_idx_0:], X1s[split_idx_1:]))
    y_test = np.hstack((y0s[split_idx_0:], y1s[split_idx_1:]))

    indices_4 = np.random.permutation(len(y_tr))
    X_tr = X_tr[indices_4]
    y_tr = y_tr[indices_4]

    return X_tr, y_tr, X_test, y_test


'''
Generate the data training set according to the training percent
'''


def trainset_percent(X, y, percent):
    number = len(X) * percent * 0.01
    number = int(number)
    X_train = X[:number]
    y_train = y[:number]
    return X_train, y_train


# Now, define the logistic regression

# Define the sigmoid function

def s_func(X, theta):
    z = -X.dot(theta)
    z = np.clip(z, -50, 50)  # clipping to avoid overflow
    g = 1 / (1 + np.exp(z))
    return g


# Define the loss function
def loss_function(X, y, theta, Lambda):
    loss = -y * np.log(s_func(X, theta)) - (1 - y) * np.log(1 - s_func(X, theta))
    total_loss = np.mean(loss, axis=0)
    regularization = Lambda / 2 * np.linalg.norm(theta) ** 2  # apply regularization rule
    return total_loss + regularization


# Define the gradient
def gradient_func(X, y, theta, Lambda):
    error = y - s_func(X, theta)
    grad = -X.T.dot(error)
    grad = grad / len(X) + Lambda * theta
    return grad


# Define the gradient descent process
def Logis_train(X, y, Lambda):
    # loss_history=np.empty(0,np.float64)
    theta = np.zeros(X.shape[1])
    for i in range(iteration):
        grad = gradient_func(X, y, theta, Lambda)
        theta = theta - LR * grad
        # loss_history=np.append(loss_history,loss_function(X,y,theta,Lambda))
    return theta


# This function is to calcuated the error of tested data
def test(X, y, theta):
    # Calculate the probability predictions
    z = -X.dot(theta)
    z = np.clip(z, -50, 50)  # avoid overflow
    prob_predictions = 1 / (1 + np.exp(z))
    # Convert probability predictions to binary class predictions (0 or 1)
    class_predictions = np.where(prob_predictions >= 0.5, 1, 0)
    # Calculate the number of mismatches
    mismatches = np.sum(class_predictions != y)
    # Compute the error rate
    error_rate = mismatches / (2 * len(y))
    return error_rate


# Define training and error testing using logistic regression.
def logisticRegression(filename, num_splits, train_percent):
    num_data = num_splits
    X, y = filereading(filename)
    X0, X1, y0, y1 = dataproecss(X, y)
    test_error1 = np.zeros((num_data, len(train_percent)))
    scaler = StandardScaler()
    # Fit the scaler on the entire dataset and transform it
    X0 = scaler.fit_transform(X0)
    X1 = scaler.transform(X1)  # Use the same scaler to transform X1

    for i in range(num_data):  # train the data 100 times
        X_tr, y_tr, Xts, yts = datasplit(X0, X1, y0, y1)
        # For every training, train the data with 5-30 percentage respectively.
        for j in range(len(train_percent)):
            tr_percent = train_percent[j]
            X_trk, y_trk = trainset_percent(X_tr, y_tr, tr_percent)
            thetak = Logis_train(X_trk, y_trk, Lambda)
            errork = test(Xts, yts, thetak)  # calculate the error
            test_error1[i][j] = errork  # store the error
        mean = np.mean(test_error1, axis=0)  # compute the error mean
        std = np.std(test_error1, axis=0)  # compute the error std
    return mean, std


# Define the function that will print out training result in the terminal
def stdout_print(mean, std, method):
    print("The following is the error mean produced by {}".format(method))
    for i in range(len(tps)):
        print("Training percent:{}%".format(tps[i]))
        print("Mean: {}".format(mean[i]))
        print("Standard Deviation: {}".format(std[i]))
        print('\n')


# Define the function that will generate the plots of error mean and std.
def graph_plot(mean, std, style, method):
    colors = ['red', 'blue', 'green', 'cyan', 'black', 'yellow']
    x = np.arange(len(mean))
    plt.plot(x, mean, marker='o', linestyle='-', color=style,
             label='Mean for {}'.format(method))
    for i, (mean_i, std_i, color) in enumerate(zip(mean, std, colors)):
        plt.fill_between([x[i], x[i]], mean_i - std_i,
                         mean_i + std_i, color='black',
                         alpha=1.0,
                         label='Std Dev for {}'.format(method) if i == 0 else "")

    plt.xticks(x, ['5%', '10%', '15%', '20%', '25%', '30%'])
    plt.ylabel('Error')
    plt.title('Means and Standard Deviations from {}'.format(method))
    plt.legend()

    # Display the plot
    plt.tight_layout()
    plt.show()


# Naive Bayes
'''
given univariate gaussian distribuion,we can calculate the conditional
probability of each xi given y P(xi|y). After getting the single probability
multiply them together to get the proability of entire x features. 

P(X|y)=P(x1,x2,x3...x57|y)=P(x1|y)P(x2|y)...P(x57|y)
'''


def conprob(X, mean, std, prob):
    std += 1e-12
    coef = 1 / (np.sqrt(2 * np.pi) * std)
    conp = coef * np.exp(-1 / 2 * (X - mean) ** 2 / (std ** 2))
    '''
    conp represents the conditional probability of each xi
    given a y
    '''
    total_conp = np.prod(conp, axis=1) * prob

    '''total_conp represents conditional probability of each 
    set of feature x given a y
    '''
    return total_conp


'''
Now, we can compute probability for y=0 and y=1. 
Using the equation P(y|X)=P(X|y)P(y)/P(X)
Sinec P(X) is the same for both case, then, we only need to compare 
P(X|y)P(y)
'''


def bayes_predic(X, y, mean0, mean1, std0, std1, py0, py1):
    # predicts probability of y=0 given x
    conprob_y0 = conprob(X, mean0, std0, py0)
    # predicts probability of y=1 given x
    conprob_y1 = conprob(X, mean1, std1, py1)
    return conprob_y0, conprob_y1


'''
Once we have the predicted result, we can compute the error using the 
true value from teh data set. 
'''


def error_cal(p0, p1, y):
    # Calculate the predicted labels: 1 if p1 is greater, else 0
    predicted_labels = np.where(p1 > p0, 1, 0)

    # Calculate the number of misclassifications
    mismatches = np.sum(predicted_labels != y)

    # Compute the error rate
    error_rate = mismatches / len(y)

    return error_rate


# Define training and error testing using Naive Bayes.
def naiveBayesGaussian(filename, num_splits, train_percent):
    X, y = filereading(filename)
    X0, X1, y0, y1 = dataproecss(X, y)
    error_store = np.zeros((num_data, len(train_percent)))
    for i in range(num_data):
        X_tr, y_tr, Xts, yts = datasplit(X0, X1, y0, y1)
        # For every training, train the data with 5-30 percentage respectively.
        for j in range(len(train_percent)):
            tr_percent = train_percent[j]
            X_trk, y_trk = trainset_percent(X_tr, y_tr, tr_percent)
            '''
 To apply the Bayes rule, firstly, we need to gather the mean, std
 of non-spam feature and spam feature repsectively. Next, we need to get
 the probability of spam and non-spam. 
 '''
            mean0 = np.zeros(X.shape[1])
            mean1 = np.zeros(X.shape[1])
            std0 = np.zeros(X.shape[1])
            std1 = np.zeros(X.shape[1])
            '''
 if in the randomly selected trainning, 
 there is no non-spam and its features
 the mean and std will be zero. 
 '''
            if X_trk[y_trk == 0].shape[0] == 0:
                # print("1")
                mean0 = np.zeros(X.shape[1])
                std0 = np.zeros(X.shape[1])
            else:
                # print("2")
                mean0 = np.mean(X_trk[y_trk == 0], axis=0)
                std0 = np.std(X_trk[y_trk == 0], axis=0)
                '''
if in the randomly selected trainning, there is no spam and its features
the mean and std will be zero. 
'''

            if X_trk[y_trk == 1].shape[0] == 0:
                # print("3")
                mean1 = np.zeros(X.shape[1])
                std1 = np.zeros(X.shape[1])
            else:
                # print("4")
                mean1 = np.mean(X_trk[y_trk == 1], axis=0)
                std1 = np.std(X_trk[y_trk == 1], axis=0)

            py1 = np.mean(y_trk)
            py0 = 1 - py1

            p0, p1 = bayes_predic(Xts, yts, mean0, mean1, std0, std1, py0, py1)
            error = error_cal(p0, p1, yts)
            error_store[i][j] = error
    mean = np.mean(error_store, axis=0)
    std = np.std(error_store, axis=0)
    return mean, std


filename = "spambase.data"
tps = [5, 10, 15, 20, 25, 30]
num_data = 100
filename = str(input("Please enter the filename\n"))
num_data = int(input("Please enter number of splits:\n"))
tr_per = input("Please enter the trainning percent:\n")

numbers = tr_per.strip("()").split(",")
tps = np.array([int(num) for num in numbers])
# convert the user input into array

LR = 0.001
iteration = 15
Lambda = 0.01
X, y = filereading(filename)
theta = np.zeros(X.shape[1])
# meanL, stdL = logisticRegression("spambase.data",100,[5,10,15,20,25,30])
# meanB, stdB = naiveBayesGaussian("spambase.data",100,[5,10,15,20,25,30])
meanL, stdL = logisticRegression(filename, num_data, tps)
meanB, stdB = naiveBayesGaussian(filename, num_data, tps)

stdout_print(meanL, stdL, 'logistic regression')
stdout_print(meanB, stdB, 'Naive Bayes')

graph_plot(meanB, stdB, 'red', 'Naive Bayes')
graph_plot(meanL, stdL, 'cyan', 'logistic regression')

# Put two graphs together
colors = ['red', 'blue', 'green', 'cyan', 'black', 'yellow']
x1 = np.arange(len(meanL))
x2 = np.arange(len(meanB))

plt.plot(x1, meanL, marker='o', linestyle='-', color='cyan',
         label='Mean for Logistic Regression')
for i, (mean_i, std_i, color) in enumerate(zip(meanL, stdL, colors)):
    plt.fill_between([x1[i], x1[i]], mean_i - std_i,
                     mean_i + std_i, color='black', alpha=1,
                     label='Std Dev for Logistic Regression' if i == 0 else "")

plt.plot(x2, meanB, marker='o', linestyle='-', color='red',
         label='Mean for Naive Bayes')
for i, (mean_i, std_i, color) in enumerate(zip(meanB, stdB, colors)):
    plt.fill_between([x2[i], x2[i]], mean_i - std_i,
                     mean_i + std_i, color='grey', alpha=1,
                     label='Std Dev for Naive Bayes' if i == 0 else "")

plt.xticks(x1, ['5%', '10%', '15%', '20%', '25%', '30%'])
plt.ylabel('Error')
plt.title('Means and Standard Deviations from two methods')
plt.legend()
plt.show() 


