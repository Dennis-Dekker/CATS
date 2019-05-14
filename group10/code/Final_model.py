#!usr/bin/env python3

# CATS group 10:
#   Myrthe van Baardwijk
#   Martin Banchero
#   Dennis Dekker
#   Marina Diachenko
# 13-05-2019

import argparse
import csv
import pickle

import pandas as pd
from numpy import ravel
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC, LinearSVC


class Model:
    def __init__(self, svm, c_l1, best_features, accuracy_validate, best_params, n_features):
        self.svm = svm
        self.c_l1 = c_l1
        self.best_features = best_features
        self.accuracy_validate = accuracy_validate
        self.best_params = best_params
        self.n_features = n_features


def predict_labels(model, test):
    """ Predicts labels on test data.

    :param model: Best model after feature and hyperparameter selection.
    :param test: Raw test data.
    :return: Saves prediction in tab separated file.
    """
    # Remove unneeded columns
    test = test.drop(["Chromosome", "Start", "End", "Nclone"], axis=1).transpose()

    # Extract only best features from test data
    x_test = model.best_features.transform(test)

    # Predict
    prediction = pd.DataFrame(model.svm.predict(x_test), columns=["Subgroup"])
    samples = pd.DataFrame(test.index, columns=["Sample"])
    final_predictions = samples.join(prediction)

    # Save predictions to .txt file
    final_predictions.to_csv("../output/predictions.txt", header=True, index=None, sep='\t', quoting=csv.QUOTE_ALL)


def l1_selection(x_train, y_train, c_li):
    """ Selects best features from data using L1 regularization and linear SVM.

    :param x_train: Raw formatted train data.
    :param y_train: Labels train data.
    :param c_li: Inverse of regularization strength, lower c mean less features.
    :return: Train data with selected features, the linear model and amount of features selected.
    """
    # Train model
    lsvc = LinearSVC(C=c_li, penalty="l1", dual=False, max_iter=100000).fit(x_train, y_train)

    # Select best features and apply filter on train data
    model = SelectFromModel(lsvc, prefit=True)
    x_new = model.transform(x_train)

    n_features = x_new.shape[1]
    print("L1: x_train amount of features from " + str(x_train.shape[1]) + " to " + str(x_new.shape[1]))

    return pd.DataFrame(x_new), model, n_features


def hyper_par_sel(x_train, y_train, estimator, param):
    """ Hyperparameter selection cross validation.

    :param x_train: Raw formatted train data.
    :param y_train: Labels train data.
    :param estimator: Method used for training, here SVM.
    :param param: Hyperparameter ranges.
    :return: Selected model based on highest validation accuracy score.
    """
    # Constants
    state = 6
    c_l1_param = [0.01, 0.1, 1, 10]
    models = []
    n_splits = 4

    # Perform hyperparameter selection on a range of C values
    for c_l1 in c_l1_param:
        # L1 feature selection
        x_train_cv, best_features, n_features = l1_selection(x_train, ravel(y_train), c_l1)

        # Split data in equal parts
        train_split = StratifiedKFold(n_splits=n_splits, shuffle=False, random_state=state)

        # Loop for hyperparameters tuning
        gscv = GridSearchCV(estimator=estimator, param_grid=param, cv=train_split, verbose=1, n_jobs=-1)

        # Train  model
        gscv.fit(x_train_cv, ravel(y_train))

        best_hyperparameters = gscv.best_params_
        val_score = gscv.best_score_

        # Save model
        model = Model(gscv, c_l1, best_features, val_score, best_hyperparameters, n_features)
        models.append(model)

    # Determine best model based on validation accuracy scores
    highest_model = models[0]
    highest_model_score = models[0].accuracy_validate
    for model in models:
        current_score = model.accuracy_validate
        if current_score > highest_model_score:
            highest_model = model
            highest_model_score = current_score

    # Extract values of best model
    best_parameters = highest_model.best_params
    c_l1 = highest_model.c_l1
    validation_score = highest_model_score
    n_features = highest_model.n_features

    print("\n",
          "Best hyper parameters:\t", best_parameters, "\n",
          "Validation score: \t", validation_score, "\n",
          "L1 C parameter: \t", c_l1, "\n",
          "Number of features: \t", n_features)

    return highest_model


def train_svm(data, labels, hyperparameters):
    """ Train SVM on whole data set.

    :param data: Raw train data.
    :param labels: Raw labels train data.
    :param hyperparameters: Hyperparameter ranges.
    :return: Selected model based on highest validation accuracy score.
    """
    # Remove unneeded columns
    df_data = data.drop(["Chromosome", "Start", "End", "Nclone"], axis=1).transpose()
    labels = labels.set_index(labels.loc[:, "Sample"]).drop("Sample", axis=1)

    best_model = hyper_par_sel(df_data, labels, SVC(), hyperparameters)

    return best_model


def load_data(input_file, label_file, test_file):
    """ Load data from files

    :param input_file: Train data file location
    :param label_file: Labels train data file location
    :param test_file: Test data file location
    :return: Pandas dataframes of train, train labels and test data.
    """
    data = pd.read_csv(input_file, sep="\t")
    labels = pd.read_csv(label_file, sep="\t")
    test = pd.read_csv(test_file, sep="\t")

    return data, labels, test


def main():
    """Main function"""

    # Parser
    parser = argparse.ArgumentParser(description='Training final SVM with L1 feature selection method.')
    parser.add_argument("-i", "--input", dest="input_file",
                        type=str, help="train data file", metavar="FILE")
    parser.add_argument("-l", "--labels", dest="label_file",
                        type=str, help="label file", metavar="FILE")
    parser.add_argument("-t", "--test", dest="test_file",
                        type=str, help="test data file", metavar="FILE")
    parser.add_argument("-o", "--output", dest="output_folder",
                        type=str, help="output folder", metavar="FOLDER")

    args = parser.parse_args()

    # If no input files are given, use default files
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

    if args.test_file is None:
        default_test_file = "../data/Validation_call.txt"
        print("No test file given, using default input file:\t" + default_test_file)
        args.test_file = default_test_file

    # hyperparameter ranges for SVM
    tuned_parameters = [{"C": [0.1, 1, 10, 100],
                         "kernel": ['linear', 'rbf', 'sigmoid', 'poly'],
                         "degree": [0, 1, 2, 3, 4],
                         "gamma": ['auto']}
                        ]

    # Load data
    data, labels, test = load_data(args.input_file, args.label_file, args.test_file)

    # Perform training
    best_model = train_svm(data, labels, tuned_parameters)

    # Predict labels test data
    predict_labels(best_model, test)

    # Save model
    with open('model.pkl', 'wb') as output:
        pickle.dump(best_model, output, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
