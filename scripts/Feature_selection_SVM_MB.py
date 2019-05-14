#!/usr/bin/env python
# coding: utf-8
import sys
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import TruncatedSVD, PCA, KernelPCA, SparsePCA
from sklearn.feature_selection import RFECV
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV,StratifiedKFold, KFold, ParameterGrid
from sklearn.metrics import classification_report,accuracy_score,precision_score,recall_score 
from sklearn import preprocessing
import matplotlib.pyplot as plt
import os



def adjust_data(data, labels):
	#Transpose
    df_x = data.transpose()
    X_data = df_x.iloc[4:,]
    y = labels["Subgroup"]
	
    #to df
    df_y = pd.DataFrame(y)

    #change categorical to numerical(classes)
    le = preprocessing.LabelEncoder()
    le.fit(np.unique(df_y.values))
    y = df_y.apply(le.transform)
    y_data= y.values.ravel()

    return X_data, y_data


def fit_SVC(X_train, y_train, X_test, y_test, best_f):
	# Set the parameters to test 
	# hyperparameter range for SVM
    tuned_parameters_outSVC = [{"C": [0.1, 1, 10, 100],
                         "kernel": ['linear', 'rbf', 'sigmoid', 'poly'],
                         "degree": [0, 1, 2, 3, 4],
                         "gamma": ['auto']}]

	#Cross validation parameters
    cv_parameters_out = KFold(n_splits=5, shuffle=True, random_state=6)

	#Hyperparameters tuning
    GSCV=GridSearchCV(estimator=SVC(), param_grid=tuned_parameters_outSVC, cv=cv_parameters_out, verbose=0,n_jobs=-1)
		
	#train a model with each set of parameters
    GSCV.fit(X_train, y_train)
    best_hyperparameters= GSCV.best_params_
		
	#Pass the best hyperparameters
    svclassifier = SVC(**best_hyperparameters)
    svclassifier.fit(X_train, y_train) 

	# Training Accuracy
    y_pred = svclassifier.predict(X_train)
    print("Training Accuracy:", accuracy_score(y_train, y_pred))

    y_pred = svclassifier.predict(X_test)  
	
    print("Testing Accuracy:",accuracy_score(y_test, y_pred))

    print("Best hyperparameters for SVC:", best_hyperparameters)


def do_RFECV(data, labels, random_state):
	
    X_data, y_data = adjust_data(data, labels)
    
    kf = KFold(n_splits = 5, shuffle=True, random_state=random_state)
    
    k_iteration = 1
    
    for train_index, test_index in kf.split(X_data):
        print("###KFold Iteration: %s" % k_iteration)
        
		# Split the dataset, 80% Train
        X_train = X_data.iloc[train_index] 
        y_train = y_data[train_index]
		
        X_test = X_data.iloc[test_index]
        y_test = y_data[test_index]


		# Set the parameters to test 
        tunned_parameters = [{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

		#Cross validation parameters
        cv_parameters = KFold(n_splits=5, shuffle=True, random_state=random_state)

		#Hyperparameters tuning
        GSCV=GridSearchCV(estimator=SVC(), param_grid=tunned_parameters, cv=cv_parameters, verbose=0,n_jobs=-1)
		
		#train a model with each set of parameters
        GSCV.fit(X_train, y_train)
        best_hyperparameters= GSCV.best_params_
        
        #Pass the best hyperparameters
        svc_RFECV = SVC(**best_hyperparameters)
        
        
		# The "accuracy" scoring is proportional to the number of correct classifications
        rfecv = RFECV(estimator = svc_RFECV, 
                        step = 1,
                        min_features_to_select = 10, 
                        cv = StratifiedKFold(5),
                        scoring = 'accuracy', n_jobs = -1)
		#fit the data
        rfecv.fit(X_train, y_train)
        
        #Optimal features
        #best_f = rfecv.get_support(1)
        #Selected features sorted by ranking(rfecv.ranking_)
        supp= rfecv.support_
        rank=rfecv.ranking_
        best_f=rank[supp[rank]] 
        
        X_train_select = X_train[X_train.columns[best_f]] # Input for the SVM
        X_test_select  = X_test[X_test.columns[best_f]] # Input for the SVM
        
        print('Original number of features :', len(X_train.columns))
        print("Number of features after elimination: %f [Acc: %.1f]" %(rfecv.n_features_, rfecv.grid_scores_[rfecv.n_features_])) 
        path = "./Features_db/"
        df_best_f = X_data[X_data.columns[best_f]].transpose()
        df_best_f.to_csv(os.path.join(path,str(k_iteration)+"-"+str(random_state)+'feature.csv'))
        
        #Fit the SVC after feature selection
        fit_SVC(X_train_select, y_train, X_test_select, y_test, best_f)
        
        k_iteration += 1
        
        print("\n")

def main(args):
    
    X = pd.read_csv("../data/Train_call.txt", delimiter="\t")
    df_feat= X.transpose()
    labels = pd.read_csv("../data/Train_clinical.txt",  delimiter="\t")
    
    #Save print() output in file
    or_stdout = sys.stdout
    file_out = open('./Features_db/Output_RFECV.txt', 'w')
    sys.stdout = file_out
    
    random_state=6#Starting with random seed 6
    
    #Loop to repeat the method 3 times
    for i in range(3):
        print("############################ Number of repeat:",i,"| Random state:",random_state,"#########################")
        if random_state > 3:#min random_state 4
            do_RFECV(X, labels, random_state)
            
        random_state= random_state-1
    return 0

    sys.stdout = or_stdout
    file_out.close()

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
