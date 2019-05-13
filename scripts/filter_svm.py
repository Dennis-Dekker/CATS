# Packages
import numpy as np
import pandas as pd
import seaborn as sns
from sys import argv
import csv
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV,StratifiedKFold,KFold, ParameterGrid, StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import TruncatedSVD, PCA, KernelPCA, SparsePCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, precision_score, recall_score
from sklearn import preprocessing
import matplotlib.pyplot as plt
import sys, time, threading


class Model():
    def __init__(self, svm, n_features, best_features, accuracy_validate, best_params):
        self.svm = svm
        self.n_features = n_features
        self.best_features = best_features
        self.accuracy_validate = accuracy_validate
        self.best_params = best_params


def preprocess_y(y):
    y = y.iloc[:,1:3]
    le = preprocessing.LabelEncoder()
    le.fit(np.unique(y.values))
    y = y.apply(le.transform)
    y = y.values.ravel()
    return y


def outer_cross_val(x, y, n_iterations):
    models = []

    kf = StratifiedKFold(n_splits=n_iterations)
    for train_index, test_index in kf.split(x, y):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        best_parameters, n_features, best_features, validation_score = inner_cross_val(x_train, y_train, 4)
        # get the selected features for the train and test set
        x_train_transformed = best_features.transform(x_train)
        x_test_transformed = best_features.transform(x_test)
        # get the best parameters
        c = best_parameters['C']
        degree = best_parameters['degree']
        gamma = best_parameters['gamma']
        kernel = best_parameters['kernel']
        svc = SVC(C=c, kernel=kernel, degree=degree, gamma=gamma)
        svc.fit(x_train_transformed, y_train)
        val_score = svc.score(x_test_transformed, y_test)
        model = Model(svc, n_features, best_features, val_score, best_parameters)
        models.append(model)
    return models


def inner_cross_val(x, y, n_iterations):
    parameter_grid = [{"C": [0.1, 1, 10, 100],
                  "kernel": ['linear','rbf','sigmoid','poly'],
                  "degree":[0, 1, 2, 3, 4],
                  "gamma": ['auto']}
                      ]
    n_feature_sizes = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
    models = []

    for n_features in n_feature_sizes:
        best_features = select_features(x, y, n_features)
        x_train = best_features.transform(x)

        cv_parameters =StratifiedKFold(n_splits=n_iterations, shuffle=True, random_state=0)
        GSCV = GridSearchCV(estimator=SVC(), param_grid=parameter_grid, cv=cv_parameters, verbose=0, n_jobs=-1)
        GSCV.fit(x_train, y)
        best_hyperparameters = GSCV.best_params_
        val_score = GSCV.best_score_

        print(n_features)
        print(best_hyperparameters)
        print(val_score)
        model = Model(GSCV, n_features, best_features, val_score, best_hyperparameters)
        models.append(model)

    highest_model = models[0]
    highest_model_score = models[0].accuracy_validate

    for model in models:
        current_score = model.accuracy_validate
        if current_score > highest_model_score:
            highest_model = model

    best_parameters = highest_model.best_params
    n_features = highest_model.n_features
    best_features = highest_model.best_features
    validation_score = highest_model_score

    return best_parameters, n_features, best_features, validation_score


def select_best_model(models):
    highest_accuracy = 0

    for i in range(len(models)):
        model = models[i]
        if model.accuracy_validate > highest_accuracy and model.accuracy_validate != 1.0:
            highest_accuracy = model.accuracy_validate
            best_model = model

    return best_model


def select_features(x_train, y_train, n_features):
    X_indices = np.arange(x_train.shape[-1])
    selector = SelectKBest(f_classif, k=n_features)
    k_best = selector.fit(x_train, y_train)
    scores = -np.log10(k_best.pvalues_)
    scores /= scores.max()
    plt.bar(X_indices - .45, scores, width=.2,
        label=r'Univariate score ($-Log(p_{value})$)', color='darkorange',
        edgecolor='black')
    #plt.show()
    return k_best


def get_best_features(feature_scores, features, n_features):
    highest_scores = []
    highest_indeces = []

    for i in range(n_features):
        highest_score = 0
        highest_index = 0
        for j in range(feature_scores.size):
            score = feature_scores[j]
            index = j
            print(score)
            if score > highest_score and index not in highest_indeces:
                highest_score = score
                highest_index = index
        highest_scores.append(highest_score)
        highest_indeces.append(highest_index)

    feature_df = features.iloc[:, highest_indeces]

    print(highest_scores)
    print(highest_indeces)
    print(np.amax(feature_scores))
    print(feature_df)
    return feature_df


def main():
    # input data files
    x = argv[1]
    y = argv[2]

    # load data
    x = pd.read_csv(x, delimiter="\t")
    y = pd.read_csv(y, delimiter="\t")
    x = x.transpose()
    x = x.iloc[4:,]
    y = preprocess_y(y)

    models = outer_cross_val(x,y,5)
    for model in models:
        print(model.accuracy_validate)
        
    best_model = select_best_model(models)
    feature_scores = best_model.best_features.scores_
    best_features = get_best_features(feature_scores, features, 10)
    print(best_features)
    print(best_model.accuracy_validate)
    best_features.to_csv('best_features_filter_svm.csv')


if __name__ == '__main__':
    main()
