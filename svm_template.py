# Starting code for HW5 SVM

import numpy as np
np.random.seed(37)
import random
import sklearn
from sklearn.svm import SVC
import scipy.stats
import matplotlib.pyplot as plt

# Dataset information
# the column names (names of the features) in the data files
# you can use this information to preprocess the features
col_names_x = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
             'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
             'hours-per-week', 'native-country']
col_names_y = ['label']

numerical_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss',
                  'hours-per-week']
categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship',
                    'race', 'sex', 'native-country']

# 1. Data loading from file and pre-processing.
# Hint: Feel free to use some existing libraries for easier data pre-processing. 
# For example, as a start you can use one hot encoding for the categorical variables and normalization 
# for the continuous variables. Also, look out for missing values. 
def load_data(csv_file_path):
    # your code here
    data = open(csv_file_path, 'r')
    dataset = []
    for line in data:
        dataset.append(line.split())

    y = np.zeros(len(dataset))
    categories = [{}]
    count = np.ones(len(dataset[0]) - 1)
    lineCount = 0
    numerics = []
    for line in dataset:
        catCount = 0
        for val in line:
            val = val[0:-1]
            if lineCount == 0 and catCount != 0:
                categories.append({})
            if lineCount == 0:
                numerics.append(val.isnumeric())
            if catCount == len(dataset[0]) - 1:
                if val == ">50":
                    y[lineCount] = 1
                else:
                    y[lineCount] = 0
                continue
            if not val.isnumeric():
                if not val in categories[catCount].keys():

                    categories[catCount][val] = len(categories[catCount]) + 1
                    count[catCount] = count[catCount] + 1
            catCount = catCount + 1
        lineCount = lineCount + 1

    count[0] = 0
    cumCount = np.cumsum(count,dtype='int16')
    #print(cumCount)
    x = np.zeros((len(dataset), cumCount[len(dataset[0]) - 2]))
    lineCount = 0
    for line in dataset:
        catCount = 0
        for val in line:
            val = val[0:-1]
            if catCount == len(dataset[0]) - 2:
                continue
            if val.isnumeric():
                # print(val,lineCount,cumCount[catCount])
                # print(x[lineCount,cumCount[catCount]])
                x[lineCount,cumCount[catCount]] = float(val)
            else:
                x[lineCount][cumCount[catCount] + categories[catCount][val]] = 1
            catCount = catCount + 1
        lineCount = lineCount + 1

    for i in range(catCount):
        if numerics[i]:
            #print(x[:,i],i)
            x[:,cumCount[i]] = scipy.stats.zscore(x[:,cumCount[i]],axis=0)
            #print(scipy.stats.zscore(x[:,i],axis=1))
            #print(scipy.stats.zscore(x[:,i], axis=0))
    return x, y   

# 2. Select best hyperparameter with cross validation and train model.
# Attention: Write your own hyper-parameter candidates.
def train_and_select_model(training_csv):
    # load data and preprocess from filename training_csv
    x_train, y_train = load_data(training_csv)
    # hard code hyperparameter configurations, an example:
    param_set = [
                 # {'kernel': 'rbf', 'C': 0.01, 'degree': 1},
                 # {'kernel': 'rbf', 'C': 0.1, 'degree': 1},
                 # {'kernel': 'rbf', 'C': 1, 'degree': 1},
                 {'kernel': 'rbf', 'C': 10, 'degree': 1},
                 # {'kernel': 'rbf', 'C': 100, 'degree': 1},
                 # {'kernel': 'linear', 'C': 0.01, 'degree': 1},
                 # {'kernel': 'linear', 'C': 0.1, 'degree': 1},
                 # {'kernel': 'linear', 'C': 1, 'degree': 1},
                 # {'kernel': 'linear', 'C': 10, 'degree': 1},
                 # {'kernel': 'linear', 'C': 100, 'degree': 1},
                 # {'kernel': 'poly', 'C': 1, 'degree': 1},
                 # {'kernel': 'poly', 'C': 1, 'degree': 3},
                 # {'kernel': 'poly', 'C': 1, 'degree': 5},
                 # {'kernel': 'poly', 'C': 1, 'degree': 7},

    ]
    # your code here
    # iterate over all hyperparameter configurations
    accuracyList = np.zeros(len(param_set))
    accuracies = [0.8295916404416676,0.8505483832253954,0.8547448543354,0.855028048295596,0.8428504455364658,0.8470470240208496,0.8481025678566906,0.848334281767249,0.84833427779042,0.8483600217920677,0.8482312918417575,0.8530199326536758,0.8445496888342187,0.8329900653963259]
    count = 0
    for param in param_set:

        numVals = x_train.shape[0]
        folds = 3
        accuracy = 0
        bestAcc = 0
        for i in range(folds):
            x_valid = x_train[int(i * numVals / folds):int((i + 1) * numVals / folds), :]
            y_valid = y_train[int(i * numVals / folds):int((i + 1) * numVals / folds)]
            x_fold = np.concatenate((x_train[0:int(i * numVals / folds), :], x_train[int((i + 1) * numVals / folds):numVals, :]))
            y_fold = np.concatenate((y_train[0:int(i * numVals / folds)], y_train[int((i + 1) * numVals / folds):numVals]))
            clf =  SVC(kernel=param['kernel'],C=param['C'],degree=param['degree'])
            clf.fit(x_fold,y_fold)
            accuracy = accuracy + clf.score(x_valid,y_valid)
        accuracy = accuracy/3
        if bestAcc < accuracy:
            bestAcc = accuracy
            bestIndex = count
        print(accuracy,param,clf.get_params())
        accuracyList[count] = accuracy
        count = count + 1
    # perform 3 FOLD cross validation
    # print cv scores for every hyperparameter and include in pdf report
    # select best hyperparameter from cv scores, retrain model
    param = param_set[bestIndex]
    clf = SVC(kernel=param['kernel'], C=param['C'], degree=param['degree'])
    best_model = clf.fit(x_train, y_train)
    best_score = bestAcc
    plt.bar([0.01,0.1,1,10,100],accuracies[0:5])
    plt.title("RBF Kernel")
    plt.xlabel("C")
    plt.ylabel("Score")
    plt.show()
    plt.figure()
    plt.bar([0.01, 0.1, 1, 10, 100], accuracies[5:10])
    plt.title("Linear Kernel")
    plt.xlabel("C")
    plt.ylabel("Score")
    plt.show()
    plt.figure()
    plt.bar([1,3,5,7], accuracies[10:14])
    plt.title("Polynomial Kernel")
    plt.xlabel("Degree")
    plt.ylabel("Score")

    return best_model, best_score

# predict for data in filename test_csv using trained model
def predict(test_csv, trained_model):
    x_test, _ = load_data(test_csv)
    predictions = trained_model.predict(x_test[:,0:114])
    return predictions

# save predictions on test data in desired format 
def output_results(predictions):
    with open('predictions.txt', 'w') as f:
        for pred in predictions:
            if pred == 0:
                f.write('<=50K\n')
            else:
                f.write('>50K\n')

if __name__ == '__main__':
    training_csv = "salary.labeled.csv"
    testing_csv = "salary.2Predict.csv"
    # fill in train_and_select_model(training_csv) to 
    # return a trained model with best hyperparameter from 3-FOLD 
    # cross validation to select hyperparameters as well as cross validation score for best hyperparameter. 
    # hardcode hyperparameter configurations as part of train_and_select_model(training_csv)
    trained_model, cv_score = train_and_select_model(training_csv)

    print("The best model was scored : ",cv_score)
    # use trained SVC model to generate predictions
    predictions = predict(testing_csv, trained_model)
    # Don't archive the files or change the file names for the automated grading.
    # Do not shuffle the test dataset
    output_results(predictions)
    # 3. Upload your Python code, the predictions.txt as well as a report to Collab.
