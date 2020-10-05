from sklearn.datasets import load_boston
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np

# self defined libraries
import LDA1dThres as LDA1dObj
import LDA2dGaussGM as LDA2dObj
import naiveBayesGaussian as GNB
import logisticRegression as LR

def bostonPercentile(percentile):
    bostonData, response = load_boston(return_X_y= True)
    classNum = np.zeros((bostonData.shape[0], 1), dtype = int)
    bostonPercentile = np.concatenate((bostonData, classNum), axis = 1)
    tPercentile = np.percentile(response, percentile)
    bostonPercentile[np.where(response >= tPercentile), 13] = 1

    return bostonPercentile[:, 0:13], bostonPercentile[:, 13]

def getDigits():
    digitsData, target = load_digits(return_X_y= True)
    target = target.reshape(digitsData.shape[0], 1)
    
    return digitsData, target

def my_cross_val(method, X, y, k):
    xdim, ydim = X.shape
    idx = np.array_split(np.random.permutation(xdim), k)
    errorRate_test = []
    errorRate_train = []

    for i in range(0, k):
        test_idx = idx[i]
        train_idx = np.array([x for x in np.arange(0, xdim) if x not in test_idx])

        # train
        train_model = method.fit(X[train_idx], y[train_idx])
        new_y = train_model.predict(X[test_idx])
        new_y_train = train_model.predict(X[train_idx])
        error_rate_test = round(np.count_nonzero(np.equal(new_y, y[test_idx]) == False)/len(new_y), 4)
        error_rate_train = round(np.count_nonzero(np.equal(new_y_train, y[train_idx]) == False)/len(new_y_train), 4)
        errorRate_test.append(error_rate_test)
        errorRate_train.append(error_rate_train)

    std_test = np.std(errorRate_test)
    std_train = np.std(errorRate_train)
    mean_test = np.mean(errorRate_test)
    mean_train = np.mean(errorRate_train)
    return errorRate_test, errorRate_train, std_test, std_train, mean_test, mean_train

def print_result_k_fold(method, X, y, k, method_name, data_name):
    errorRate_test, errorRate_train, std_test, std_train, mean_test, mean_train = my_cross_val(method, X, y, k)
    print(f"{data_name} with {method_name}:\nperformance on testing data:")

    for i in range(k):
        print(f"fold {str(i)}: {errorRate_test[i]}")

    print(f"mean: {mean_test}\nstandard deviation: {std_test}\n")

    print("performance on training data:")
    for i in range(k):
        print(f"fold {str(i)}: {errorRate_train[i]}")

    print(f"mean: {mean_train}\nstandard deviation: {std_train}\n")

def custom_weighted_split(X, pi):
    n = X.shape[0]
    train_idx = np.random.choice(n, int(n*pi), replace = False)
    test_idx = np.array([x for x in np.arange(0, n) if x not in train_idx])

    return train_idx, test_idx

def ce(y, new_y):
    eps = 1e-15
    new_y = np.clip(new_y, eps, 1 - eps)
    new_y = np.where(new_y == 0, 1e-9, new_y)
    return -np.sum(y*np.log(new_y))/y.shape[0]

def ee(y, new_y):
    y = np.argmax(y, axis=1).ravel()
    new_y = np.argmax(new_y, axis=1).ravel()
    return round(np.count_nonzero(np.equal(new_y, y) == False)/len(new_y), 4)

def my_train_test(method, X, y, k, num_splits, train_percent, cross_entropy = False):
    errorRate_test = np.array([]).reshape((0, len(train_percent)))
    train_idx = []
    test_idx = []
    temp_train_idx, temp_test_idx = custom_weighted_split(X, num_splits)
    test_incre = np.array_split(temp_train_idx, k-1)

    for i in range(k):
        if i > 0:
            temp_test_idx = np.append(temp_test_idx[len(test_incre[i-1]):], test_incre[i-1], axis=0)
            temp_train_idx = np.array([x for x in np.arange(0, X.shape[0]) if x not in temp_test_idx])

        test_idx = temp_test_idx
        error_rate_test = np.array([])
        error_rate_train = np.array([])
        error_rate_test_temp = np.array([])

        for percent in train_percent:
            train_idx = temp_train_idx[: int(temp_train_idx.shape[0]*percent/100)]

            # train
            train_model = method.fit(X[train_idx], y[train_idx])
            new_y = train_model.predict(X[test_idx])
            if cross_entropy:
                error_rate_test = np.append(error_rate_test, ee(y[test_idx],new_y))
            else:
                error_rate_test = np.append(error_rate_test, round(np.count_nonzero(np.equal(new_y, y[test_idx]) == False)/len(new_y), 4))

        errorRate_test = np.vstack((errorRate_test, error_rate_test))

    mean_test = np.mean(errorRate_test, axis = 0)
    std_test = np.std(errorRate_test, axis = 0)

    return errorRate_test, std_test, mean_test

def print_result_weighted(method, X, y, k, num_split, train_percentage, method_name, data_name, cross_entropy=False):
    errorRate_test, std_test, mean_test =my_train_test(method, X, y, k, num_splits, train_percentage,cross_entropy)

    print(f"{data_name} with {method_name}:\nperformance on testing data:")

    for j in range(len(train_percentage)):
        print(f"{train_percentage[j]}% of training data:")
        for i in range(k):
            print(f"fold {str(i)}: {errorRate_test[i,j]}")

        print(f"mean: {mean_test[j]}\nstandard deviation: {std_test[j]}\n")

    return np.asarray(mean_test), np.asarray(std_test)

def LDA1dThres(num_crossval):
    Boston50X, Boston50y = bostonPercentile(50)
    print_result_k_fold(LDA1dObj.LDA1dThres(), Boston50X, Boston50y.ravel(), num_crossval, "LDA1dThres", "Boston50")

def LDA2dGaussGM(num_crossval):
    DigitsX, Digitsy = getDigits()
    print_result_k_fold(LDA2dObj.LDA2dGaussGM(10,2), DigitsX, Digitsy.ravel(), num_crossval, "LDA2dGaussGM", "Digits")

def naiveBayesGaussian(num_splits, train_percent):
    Boston50X, Boston50y = bostonPercentile(50)
    Boston75X, Boston75y = bostonPercentile(75)
    DigitsX, Digitsy = getDigits()

    LR_Digitsy = np.zeros((Digitsy.shape[0], 10))
    for i in range(LR_Digitsy.shape[0]):
        LR_Digitsy[i, Digitsy[i,0]] = 1

    LR_Boston50y = np.zeros((Boston50y.shape[0], 2))
    for i in range(LR_Boston50y.shape[0]):
        LR_Boston50y[i, int(Boston50y[i])] = 1

    LR_Boston75y = np.zeros((Boston75y.shape[0], 2))
    for i in range(LR_Boston75y.shape[0]):
        LR_Boston75y[i, int(Boston75y[i])] = 1

    error_digits, std_digits = print_result_weighted(GNB.naiveBayesGaussian(10),  DigitsX, Digitsy.ravel(), 10, num_splits, train_percent, "naiveBayesGaussian", "Digits")
    error_boston50, std_50 = print_result_weighted(GNB.naiveBayesGaussian(2),  Boston50X, Boston50y.ravel(), 10, num_splits, train_percent, "naiveBayesGaussian", "Boston50")
    error_boston75, std_75  = print_result_weighted(GNB.naiveBayesGaussian(2),  Boston75X, Boston75y.ravel(), 10, num_splits, train_percent, "naiveBayesGaussian", "Boston75")

    return np.array([error_digits, error_boston50, error_boston75]), np.array([std_digits, std_50, std_75])

def logisticRegression(num_splits, train_percent):
    Boston50X, Boston50y = bostonPercentile(50)
    Boston75X, Boston75y = bostonPercentile(75)
    DigitsX, Digitsy = getDigits()

    LR_Digitsy = np.zeros((Digitsy.shape[0], 10))
    for i in range(LR_Digitsy.shape[0]):
        LR_Digitsy[i, Digitsy[i,0]] = 1

    LR_Boston50y = np.zeros((Boston50y.shape[0], 2))
    for i in range(LR_Boston50y.shape[0]):
        LR_Boston50y[i, int(Boston50y[i])] = 1

    LR_Boston75y = np.zeros((Boston75y.shape[0], 2))
    for i in range(LR_Boston75y.shape[0]):
        LR_Boston75y[i, int(Boston75y[i])] = 1
    
    error_digits, std_digits = print_result_weighted(LR.logisticRegression(10, DigitsX.shape[1]),  DigitsX, LR_Digitsy, 10, num_splits, train_percent, "LogisticRegression", "Digits", True)
    error_boston50, std_50 = print_result_weighted(LR.logisticRegression(2, Boston50X.shape[1]),  Boston50X, LR_Boston50y, 10, num_splits, train_percent, "LogisticRegression", "Boston50", True)
    error_boston75, std_75 = print_result_weighted(LR.logisticRegression(2, Boston75X.shape[1]),  Boston75X, LR_Boston75y, 10, num_splits, train_percent, "LogisticRegression", "Boston75", True)    

    return np.array([error_digits, error_boston50, error_boston75]), np.array([std_digits, std_50, std_75])

def plot(x, y1, y2, y1_err, y2_err):
    # digits
    fig, ax1 = plt.subplots()
    ax1.errorbar(x, y1[0], yerr=y1_err[0], linestyle='dashed', color='blue', label = 'GNB')
    ax1.errorbar(x, y2[0], yerr=y2_err[0], color='black', label='Logistic Regression')
    ax1.set_title('digits')

    ax1.legend(loc ='upper left')
    fig.savefig("digits.png")


    # boston50
    fig, ax = plt.subplots()
    ax.errorbar(x, y1[1], yerr=y1_err[1], linestyle='dashed', color='blue', label = 'GNB')
    ax.errorbar(x, y2[1], yerr=y2_err[1], color='black', label='Logistic Regression')
    ax.set_title('Boston 50')

    ax.legend(loc ='upper left') 

    fig.savefig("Boston50.png")

    # boston75
    fig, ax = plt.subplots()
    ax.errorbar(x, y1[2], yerr=y1_err[2], linestyle='dashed', color='blue', label = 'GNB')
    ax.errorbar(x, y2[2], yerr=y2_err[2], color='black', label='Logistic Regression')
    ax.set_title('Boston 75')

    ax.legend(loc ='upper left') 

    fig.savefig("Boston75.png")


if __name__ == "__main__":
    num_splits = 0.8
    train_percentage = [10, 25, 50, 75, 100]

    LDA1dThres(10)
    LDA2dGaussGM(10)
    error_lr, std_lr = logisticRegression(num_splits, train_percentage)
    error_gnb, std_gnb = naiveBayesGaussian(num_splits, train_percentage)

    plot(train_percentage, error_gnb, error_lr, std_gnb, std_lr)