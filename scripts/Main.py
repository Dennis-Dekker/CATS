#!/usr/bin/env python3

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy import ravel
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.svm import SVC, LinearSVC


def l1_selection(x_train, y_train):
    lsvc = LinearSVC(C=1, penalty="l1", dual=False).fit(x_train, y_train)
    model = SelectFromModel(lsvc, prefit=True)
    x_new = model.transform(x_train)
    print("L1: x_train amount of features form " + str(x_train.shape[1]) + " to " + str(x_new.shape[1]))

    return pd.DataFrame(x_new)


def nested_cv(x_train, y_train, estimator, param, use_stratified):
    state = 1
    out_scores = []
    in_winner_param = []
    out_n_splits = 5
    in_n_splits = 4

    x_train = l1_selection(x_train, ravel(y_train))

    if use_stratified:
        # StratifiedKFold = Equal amount of each class per fold (uncomment next line to use)
        out_cv = StratifiedKFold(n_splits=out_n_splits, shuffle=True, random_state=state)
        print('Using stratified k-fold splitting, number of outer splits:\t' + str(out_n_splits))
    else:
        # Kfold = random amount of each class per fold
        out_cv = KFold(n_splits=out_n_splits, shuffle=True, random_state=state)
        print('Using unstratified k-fold splitting, number of outer splits:\t' + str(out_n_splits))

    for i, (index_train_out, index_test_out) in enumerate(out_cv.split(x_train, y_train)):
        x_train_out, x_test_out = x_train.iloc[index_train_out], x_train.iloc[index_test_out]
        y_train_out, y_test_out = y_train.iloc[index_train_out], y_train.iloc[index_test_out]

        in_cv = StratifiedKFold(n_splits=in_n_splits, shuffle=True, random_state=state)
        # inner loop for hyperparameters tuning
        gscv = GridSearchCV(estimator=estimator, param_grid=param, cv=in_cv, verbose=1, n_jobs=-1)
        # train a model with each set of parameters
        gscv.fit(x_train_out, ravel(y_train_out))
        # predict using the best set of hyperparameters
        prediction = gscv.predict(x_test_out)
        in_winner_param.append(gscv.best_params_)
        out_scores.append(accuracy_score(prediction, y_test_out))
        print("\nBest inner accuracy of fold " + str(i + 1) + ": " + str(gscv.best_score_) + "\n")

    for i in zip(in_winner_param, out_scores):
        print(i)
    print("Mean of outer loop: " + str(np.mean(out_scores)) + " std: " + str(np.std(out_scores)))
    return out_scores


def apply_feature_selection(x_train, y_train, feature_selection):
    pass


def nested_svm(data, labels, hyperparameters, use_stratified, feature_selection):
    print("Amount of features " + str(data.shape[0]))

    df_data = data.drop(["Chromosome", "Start", "End", "Nclone"], axis=1).transpose()
    labels = labels.set_index(labels.loc[:, "Sample"]).drop("Sample", axis=1)
    print(df_data.shape)

    # working on feature selection integration.
    # df_data = apply_feature_selection(x_train, y_train, feature_selection)

    SVM_dist = nested_cv(df_data, labels, SVC(), hyperparameters, use_stratified)


def calculate_pca(data, labels):
    # remove annotation from data except sample name
    df_data = data.drop(["Chromosome", "Start", "End", "Nclone"], axis=1).transpose()
    labels = labels.set_index(labels.loc[:, "Sample"]).drop("Sample", axis=1)

    # PCA
    pca = PCA(n_components=3)
    principal_components = pca.fit_transform(df_data)
    principal_df = pd.DataFrame(data=principal_components, columns=['principal component 1', 'principal component 2',
                                                                    'principal component 3'])
    # put sample name back
    principal_df = principal_df.set_index(df_data.index.values)
    # add disease type
    final_df = pd.concat([principal_df, labels[["Subgroup"]]], ignore_index=False, axis=1)

    # plot PC1 vs PC2
    pca_color = sns.pairplot(x_vars=["principal component 1"], y_vars=["principal component 2"], data=final_df,
                             hue="Subgroup", height=5)
    path_pca_figure_color = "../images/PCA_color.png"
    pca_color.set(xlabel="PC1 (" + str(round(pca.explained_variance_ratio_[0] * 100, 1)) + "%)")
    pca_color.set(ylabel="PC2 (" + str(round(pca.explained_variance_ratio_[1] * 100, 1)) + "%)")
    plt.show()
    pca_color.savefig(path_pca_figure_color)
    print("Image saved to: " + path_pca_figure_color)


def load_data(input_file, label_file):
    data = pd.read_csv(input_file, sep="\t")
    labels = pd.read_csv(label_file, sep="\t")
    return data, labels


def resample_data():
    print("resampled... (not implemented)")
    return


def main():
    """Main function"""

    # Parser
    parser = argparse.ArgumentParser(description='Test differnt feature selection and machine learning methods')
    parser.add_argument('feature_selection_method', choices=['No', 'ANOVA', 'L1', 'RFECV', 'all'],
                        help='Feature selection method to use')
    parser.add_argument('-r', '--resample', action="store_true", help='resample test/train data')
    parser.add_argument("-i", "--input", dest="input_file",
                        type=str, help="train data file", metavar="FILE")
    parser.add_argument("-l", "--labels", dest="label_file",
                        type=str, help="label file", metavar="FILE")
    parser.add_argument("-o", "--output", dest="output_folder",
                        type=str, help="output folder", metavar="FOLDER")
    parser.add_argument("-PCA", action="store_true", help="show pca of data")
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

    if args.resample:
        resample_data()

    # hyperparameter range for SVM
    tuned_parameters = [
        {'kernel': ['rbf'], 'gamma': ["auto", 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6], 'C': range(1, 500, 10)},
        {'kernel': ['linear'], 'C': range(1, 500, 10)},
        {'kernel': ["sigmoid"], "C": range(1, 500, 10), 'gamma': ['auto', 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]}
    ]

    # Load data 
    data, labels = load_data(args.input_file, args.label_file)

    # Simple pca
    if args.PCA:
        calculate_pca(data, labels)

    nested_svm(data, labels, tuned_parameters, args.use_unstratified, args.feature_selection_method)


if __name__ == '__main__':
    main()
