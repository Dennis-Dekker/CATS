#!/usr/bin/env python3

import glob
from itertools import cycle
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
import seaborn as sns
import sklearn
from sklearn.metrics import accuracy_score, auc, confusion_matrix, cohen_kappa_score, f1_score
import argparse
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.svm import SVC
from sklearn import svm
from numpy import ravel
from sklearn.model_selection import GridSearchCV,cross_val_score,KFold


def calculate_SVM(data, labels):
    df_data = data.drop(["Chromosome", "Start", "End", "Nclone"], axis = 1).transpose()
    labels = labels.set_index(labels.loc[:,"Sample"]).drop("Sample", axis = 1)
    
    clf = svm.SVC(gamma="scale", decision_function_shape = "ovo")
    clf.fit(df_data, ravel(labels))
    
    return
    
def nested_CV(X_train,y_train, estimator, param):
    state=1
    out_scores=[]
    in_winner_param=[]
    out_cv = KFold(n_splits=5, shuffle=True, random_state=state)
    for i, (index_train_out, index_test_out) in enumerate(out_cv.split(X_train)):
        X_train_out, X_test_out = X_train.iloc[index_train_out], X_train.iloc[index_test_out]
        y_train_out, y_test_out = y_train.iloc[index_train_out], y_train.iloc[index_test_out]

        in_cv =KFold(n_splits=4, shuffle=True, random_state=state)
        #inner loop for hyperparameters tuning
        GSCV=GridSearchCV(estimator=estimator, param_grid=param, cv=in_cv, verbose=2,n_jobs=-1)
        #train a model with each set of parameters
        GSCV.fit(X_train_out, ravel(y_train_out))
        #predict using the best set of hyperparameters
        prediction=GSCV.predict(X_test_out)
        in_winner_param.append(GSCV.best_params_)
        out_scores.append(accuracy_score(prediction, y_test_out))
        print("\nBest inner accuracy of fold "+str(i+1)+": "+str(GSCV.best_score_)+"\n")

    for i in zip(in_winner_param, out_scores):
        print(i)
    print("Mean of outer loop: "+str(np.mean(out_scores))+" std: "+str(np.std(out_scores)))
    return out_scores
    
def nested_SVM(data,labels):
    df_data = data.drop(["Chromosome", "Start", "End", "Nclone"], axis = 1).transpose()
    labels = labels.set_index(labels.loc[:,"Sample"]).drop("Sample", axis = 1)
    
    print(df_data.iloc[0:5,0:5])
        
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4, 1e-5, 1e-6], 'C': range(10,1000,10)},
    {'kernel': ['linear'], 'C': range(1,101,10)}
    ]
    SVM_dist = nested_CV(df_data,labels, SVC(), tuned_parameters)

def calculate_pca(data,labels):
    
    # remove annotation from data except sample name
    df_data = data.drop(["Chromosome", "Start", "End", "Nclone"], axis = 1).transpose()
    labels = labels.set_index(labels.loc[:,"Sample"]).drop("Sample", axis = 1)
    
    #PCA
    pca = PCA(n_components=3)
    principalComponents = pca.fit_transform(df_data)
    principalDf = pd.DataFrame(data = principalComponents
                  , columns = ['principal component 1', 'principal component 2', 'principal component 3'])
    # put sample name back
    principalDf = principalDf.set_index(df_data.index.values)
    # add disease type
    finalDf = pd.concat([principalDf, labels[["Subgroup"]]],ignore_index=False, axis = 1)
    
    pca_color=sns.pairplot(x_vars=["principal component 1"], y_vars=["principal component 2"], data=finalDf, hue="Subgroup", height=5)
    path_PCA_figure_color = "../images/PCA_color.png"
    pca_color.set(xlabel = "PC1 (" + str(round(pca.explained_variance_ratio_[0]*100, 1)) + "%)")
    pca_color.set(ylabel = "PC2 (" + str(round(pca.explained_variance_ratio_[1]*100, 1)) + "%)")
    plt.show()
    pca_color.savefig(path_PCA_figure_color)
    print("Image saved to: " + path_PCA_figure_color)
    

def load_data(input_file, label_file):
    data = pd.read_csv(input_file, sep = "\t")
    labels = pd.read_csv(label_file, sep = "\t")
    return data, labels

def resample_data():
    print("resampled... (not implemented)")
    return

def main():
    """Main function"""
    
    # Parser
    parser = argparse.ArgumentParser(description='Test differnt feature selection and machine learning methods')
    parser.add_argument('-r', '--resample', action="store_true", help='resample test/train data')
    parser.add_argument("-i", "--input", dest="input_file",
                        type=str, help="train data file", metavar="FILE")
    parser.add_argument("-l", "--labels", dest="label_file",
                        type=str, help="label file", metavar="FILE")
    parser.add_argument("-o", "--output", dest="output_folder",
                        type=str, help="output folder", metavar="FOLDER")
    parser.add_argument("-PCA", action="store_true", help="show pca of data")
                        
    args = parser.parse_args()
    
    if args.input_file == None:
        default_input_file = "../data/Train_call.txt"
        print("No input file given, using default input file:\t\t" + default_input_file)
        args.input_file = default_input_file
        
    if args.label_file == None:
        default_label_file = "../data/Train_clinical.txt"
        print("No label file given, using default input file:\t\t" + default_label_file)
        args.label_file = default_label_file
        
    if args.output_folder == None:
        default_output_folder = "../output/"
        print("No output folder given, using default input file:\t" + default_output_folder)
        args.output_folder = default_output_folder
        
    if args.resample:
        resample_data()
    
    # Load data 
    data, labels = load_data(args.input_file, args.label_file)
    
    # Simple pca
    if args.PCA:
        calculate_pca(data, labels)
    
    nested_SVM(data,labels)

if __name__ == '__main__':
    main()
