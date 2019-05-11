#!/usr/bin/env python3

import argparse

import pandas as pd
from numpy import ravel
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.svm import SVC, LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt


class Model:
    def __init__(self, svm, c_l1, best_features, accuracy_validate, best_params, n_features, iteration):
        self.svm = svm
        self.c_l1 = c_l1
        self.best_features = best_features
        self.accuracy_validate = accuracy_validate
        self.best_params = best_params
        self.n_features = n_features
        self.iteration = iteration


def l1_selection(x_train, y_train, c_li):
    lsvc = LinearSVC(C=c_li, penalty="l1", dual=False).fit(x_train, y_train)
    model = SelectFromModel(lsvc, prefit=True)
    x_new = model.transform(x_train)
    n_features = x_new.shape[1]
    print("L1: x_train amount of features form " + str(x_train.shape[1]) + " to " + str(x_new.shape[1]))

    return pd.DataFrame(x_new), model, n_features


def inner_cv(x_train, y_train, in_n_splits, state, estimator, param, iteration):
    c_l1_param = [0.01, 0.1, 1, 10]
    models = []
    for c_l1 in c_l1_param:
        x_train_in, best_features, n_features = l1_selection(x_train, ravel(y_train), c_l1)

        in_cv = StratifiedKFold(n_splits=in_n_splits, shuffle=False, random_state=state)
        # inner loop for hyperparameters tuning
        gscv = GridSearchCV(estimator=estimator, param_grid=param, cv=in_cv, verbose=1, n_jobs=-1)

        # train a model with each set of parameters
        gscv.fit(x_train_in, ravel(y_train))
        best_hyperparameters = gscv.best_params_
        val_score = gscv.best_score_

        model = Model(gscv, c_l1, best_features, val_score, best_hyperparameters, n_features, iteration)
        models.append(model)
    highest_model = models[0]
    highest_model_score = models[0].accuracy_validate

    for model in models:
        current_score = model.accuracy_validate
        if current_score > highest_model_score:
            highest_model = model
            highest_model_score = current_score

    best_parameters = highest_model.best_params
    c_l1 = highest_model.c_l1
    best_features = highest_model.best_features
    validation_score = highest_model_score

    print(best_parameters, validation_score)

    return best_parameters, c_l1, best_features, validation_score


def outer_cv(x_train, y_train, estimator, param, use_stratified):
    states = [6,5,4]
    out_n_splits = 5
    in_n_splits = 4
    models = []
    plot_number = 1
    iterations = 15

    for iteration in range(3):
        state = states[iteration]
        if use_stratified:
            # StratifiedKFold = Equal amount of each class per fold (uncomment next line to use)
            out_cv = StratifiedKFold(n_splits=out_n_splits, shuffle=False, random_state=state)
            print('Using stratified k-fold splitting, number of outer splits:\t' + str(out_n_splits))
        else:
            # Kfold = random amount of each class per fold
            out_cv = KFold(n_splits=out_n_splits, shuffle=False, random_state=state)
            print('Using unstratified k-fold splitting, number of outer splits:\t' + str(out_n_splits))

        for i, (index_train_out, index_test_out) in enumerate(out_cv.split(x_train, y_train)):
            x_train_out, x_test_out = x_train.iloc[index_train_out], x_train.iloc[index_test_out]
            y_train_out, y_test_out = y_train.iloc[index_train_out], y_train.iloc[index_test_out]

            best_parameters, c_l1, best_features, validation_score = inner_cv(x_train_out, y_train_out, in_n_splits, state,
                                                                              estimator, param, iteration)

            x_train_out_transformed = best_features.transform(x_train_out)
            x_test_out_transformed = best_features.transform(x_test_out)

            c = best_parameters["C"]
            degree = best_parameters['degree']
            gamma = best_parameters['gamma']
            kernel = best_parameters['kernel']

            svc = SVC(C=c, kernel=kernel, degree=degree, gamma=gamma)
            svc.fit(x_train_out_transformed, y_train_out)
            val_score = svc.score(x_test_out_transformed, ravel(y_test_out))
            model = Model(svc, c_l1, best_features, val_score, best_parameters, x_test_out_transformed.shape[1], iteration)
            models.append(model)

            classes = ["HER2+", "HR+", "Triple Neg"]
            y_train_out = label_binarize(y_train_out, classes=classes)
            y_test_out = label_binarize(y_test_out, classes=classes)
            n_classes = y_train_out.shape[1]
            classifier = OneVsRestClassifier(SVC(kernel=model.best_params["kernel"], degree=model.best_params["degree"],
                                                 C=model.best_params["C"], gamma=model.best_params["gamma"],
                                                 probability=True, random_state=6))
            y_score = classifier.fit(x_train_out_transformed,y_train_out).decision_function(x_test_out_transformed)

            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_out[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], _ = roc_curve(y_test_out.ravel(), y_score.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

            plt.figure()
            lw = 2
            plt.plot(fpr[0], tpr[0], color='darkorange',
                     lw=lw, label='%s (AUC = %0.2f)' % (classes[0], roc_auc[0]))
            plt.plot(fpr[1], tpr[1], color='red',
                     lw=lw, label='%s (AUC = %0.2f)' % (classes[1],roc_auc[1]))
            plt.plot(fpr[2], tpr[2], color='blue',
                     lw=lw, label='%s (AUC = %0.2f)' % (classes[2],roc_auc[2]))

            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC plot L1 ')
            plt.legend(loc="lower right")
            plt.savefig("../images/L1_feature_selection/ROC_plot%s.png" % plot_number)
            plot_number += 1

    return models


def nested_svm(data, labels, hyperparameters, use_stratified):
    print("Amount of features " + str(data.shape[0]))

    df_data = data.drop(["Chromosome", "Start", "End", "Nclone"], axis=1).transpose()
    labels = labels.set_index(labels.loc[:, "Sample"]).drop("Sample", axis=1)
    print(df_data.shape)

    models = outer_cv(df_data, labels, SVC(), hyperparameters, use_stratified)
    for model in models:
        print(model.n_features, model.best_params, model.c_l1, model.accuracy_validate, model.iteration)

    highest_model = models[0]
    highest_model_score = models[0].accuracy_validate

    for model in models:
        current_score = model.accuracy_validate
        if current_score > highest_model_score:
            highest_model = model
            highest_model_score = current_score

    best_features = highest_model.best_features
    validation_score = highest_model_score

    print(best_features.get_support(), validation_score)


def load_data(input_file, label_file):
    data = pd.read_csv(input_file, sep="\t")
    labels = pd.read_csv(label_file, sep="\t")
    return data, labels


def main():
    """Main function"""

    # Parser
    parser = argparse.ArgumentParser(description='Test differnt feature selection and machine learning methods')
    parser.add_argument("-i", "--input", dest="input_file",
                        type=str, help="train data file", metavar="FILE")
    parser.add_argument("-l", "--labels", dest="label_file",
                        type=str, help="label file", metavar="FILE")
    parser.add_argument("-o", "--output", dest="output_folder",
                        type=str, help="output folder", metavar="FOLDER")
    parser.add_argument("-unStrat", "--use_unstratified", action="store_false",
                        help="use unstratified k-fold splitting")

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

    nested_svm(data, labels, tuned_parameters, args.use_unstratified)


if __name__ == '__main__':
    main()
