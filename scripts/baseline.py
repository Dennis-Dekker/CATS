#!/usr/bin/env python3


import argparse
import pandas as pd
import numpy as np
from numpy import ravel
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.svm import SVC, LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from scipy import stats
import math


class Model:

    def __init__(self, svm, accuracy_validate, best_params):
        self.svm = svm
        self.accuracy_validate = accuracy_validate
        self.best_params = best_params


def inner_cv(x_train, y_train, in_n_splits, state, estimator, param):
    """ Inner cross validation """


    models = []

    # Use stratified KFold CV
    in_cv = StratifiedKFold(n_splits=in_n_splits, shuffle=True, random_state=state)

    # inner loop for hyperparameters tuning
    gscv = GridSearchCV(estimator=estimator, param_grid=param, cv=in_cv, verbose=1, n_jobs=-1)

    # train a model with best hyperparameters
    gscv.fit(x_train, ravel(y_train))
    best_hyperparameters = gscv.best_params_
    val_score = gscv.best_score_
    model = Model(gscv, val_score, best_hyperparameters)
    models.append(model)

    return best_hyperparameters, val_score


def outer_cv(x_train, y_train, estimator, param, state):
    """ Outer cross validation """

    out_n_splits = 5
    in_n_splits = 4
    plot_number = 1
    models = []

    # Use stratified KFold CV
    out_cv = StratifiedKFold(n_splits=out_n_splits, shuffle=True, random_state=state)
    print('Using stratified k-fold splitting, number of outer splits:\t' + str(out_n_splits))

    for i, (index_train_out, index_test_out) in enumerate(out_cv.split(x_train, y_train)):
        x_train_out, x_test_out = x_train.iloc[index_train_out], x_train.iloc[index_test_out]
        y_train_out, y_test_out = y_train.iloc[index_train_out], y_train.iloc[index_test_out]

        best_parameters, validation_score = inner_cv(x_train_out, y_train_out, in_n_splits, state,
                                                 estimator, param)

        c = best_parameters["C"]
        degree = best_parameters['degree']
        gamma = best_parameters['gamma']
        kernel = best_parameters['kernel']

        svc = SVC(C=c, kernel=kernel, degree=degree, gamma=gamma)
        svc.fit(x_train_out, y_train_out)
        val_score = svc.score(x_test_out, ravel(y_test_out))
        model = Model(svc, val_score, best_parameters)
        models.append(model)

    return models


def nested_svm(data, labels, hyperparameters, state):
    accuracies = []
    hyperparams = []
    print("Amount of features " + str(data.shape[0]))
    df_data = data.drop(["Chromosome", "Start", "End", "Nclone"], axis=1).transpose()
    labels = labels.set_index(labels.loc[:, "Sample"]).drop("Sample", axis=1)
    print(df_data.shape)

    models = outer_cv(df_data, labels, SVC(), hyperparameters, state)

    for model in models:
        accuracies.append(model.accuracy_validate)
        hyperparams.append(model.best_params)

    return accuracies, hyperparams


def load_data(input_file, label_file):
    data = pd.read_csv(input_file, sep="\t")
    labels = pd.read_csv(label_file, sep="\t")
    return data, labels


def main():
    """Main function"""
    # Parser
    parser = argparse.ArgumentParser(description='Baseline SVM classifier')
    parser.add_argument("-i", "--input", dest="input_file",
                        type=str, help="train data file", metavar="FILE")
    parser.add_argument("-l", "--labels", dest="label_file",
                        type=str, help="label file", metavar="FILE")
    parser.add_argument("-o", "--output", dest="output_folder",
                        type=str, help="output folder", metavar="FOLDER")

    args = parser.parse_args()

    if args.input_file is None:
        default_input_file = "../data/Train_call.txt"
        print("No input file given, using default input file:\t\t" + default_input_file)
        args.input_file = default_input_file

    if args.label_file is None:
        default_label_file = "../data/Train_clinical.txt"
        print("No label file given, using default input file:\t\t" + default_label_file)
        args.label_file = default_label_file

    if args.output_folder is None:
        default_output_folder = "../output/"
        print("No output folder given, using default input file:\t" + default_output_folder)
        args.output_folder = default_output_folder

    # hyperparameter range for SVM
    tuned_parameters = [{"C": [0.1, 1, 10, 100],
                         "kernel": ['linear', 'rbf', 'sigmoid', 'poly'],
                         "degree": [0, 1, 2, 3, 4],
                         "gamma": ['auto']}
                        ]

    # Load data
    data, labels = load_data(args.input_file, args.label_file)

    # 15 iterations - outer CV loop (call nested SVM).
    # Following random states for splitting will be used: 6, 5, 4
    print("1st split in 5")
    accuracies1, hyperparameters1, = nested_svm(data, labels, tuned_parameters, 6)
    # print(hyperparameters1, accuracies1)
    print("2nd split in 5")
    accuracies2, hyperparameters2 = nested_svm(data, labels, tuned_parameters, 5)
    # print(hyperparameters2, accuracies2)
    print("3rd split in 5")
    accuracies3, hyperparameters3 = nested_svm(data, labels, tuned_parameters, 4)
    # print(hyperparameters3, accuracies3)

    accuracies = accuracies1 + accuracies2 + accuracies3

    mean_acc = np.mean(accuracies)
    st_dev = np.std(accuracies)
    semAcc = stats.sem(accuracies)
    print("mean", mean_acc, "std", st_dev)
    print("SEM", semAcc)


if __name__ == '__main__':
    main()