#!/usr/bin/env python
# coding: utf-8

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
from tqdm import tqdm




def adjust_data(data, labels):
	#df_data_call = pd.read_csv("Data/Train_call.txt", delimiter="\t")
	df_x = data.transpose()
	X_data = df_x.iloc[4:,]


	#df_data_clinical = pd.read_csv("Data/Train_clinical.txt",  delimiter="\t")
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
	
	svclassifier = SVC(kernel='linear')  
	svclassifier.fit(X_train, y_train) 

	# Training Accuracy
	y_pred = svclassifier.predict(X_train)
	print("Training Accuracy:", accuracy_score(y_train, y_pred))

	y_pred = svclassifier.predict(X_test)  
	
	print("Testing Accuracy:",accuracy_score(y_test, y_pred))




def do_RFECV(data, labels):
	
	X_data, y_data = adjust_data(data, labels)
	

	
	kf = KFold(n_splits = 5, shuffle=True, random_state=0)
	
	k_iteration = 1
	
	for train_index, test_index in kf.split(X_data):
		print("KFold Iteration: %s" % k_iteration)
		# Split the dataset, 80% Train
		X_train = X_data.iloc[train_index] 
		y_train = y_data[train_index]
		
		X_test = X_data.iloc[test_index]
		y_test = y_data[test_index]


		# Set the parameters to test 
		tunned_parameters = [{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

		#Cross validation parameters
		cv_parameters = KFold(n_splits=5, shuffle=True, random_state=0)

		#Hyperparameters tuning
		GSCV=GridSearchCV(estimator=SVC(), param_grid=tunned_parameters, cv=cv_parameters, verbose=0,n_jobs=-1)
		
		#train a model with each set of parameters
		GSCV.fit(X_train, y_train)
		best_hyperparameters= GSCV.best_params_
		
		svc = SVC(**best_hyperparameters)



		# The "accuracy" scoring is proportional to the number of correct classifications
		rfecv = RFECV(	estimator = svc, 
						step = 1,
						min_features_to_select = 5, 
						cv = StratifiedKFold(5),
						scoring = 'accuracy', n_jobs = -1)


		#fit the data
		rfecv.fit(X_train, y_train)


		best_f = rfecv.get_support(1)
		X_train_select = X_train[X_train.columns[best_f]] # Input for the SVM
		X_test_select  = X_test[X_test.columns[best_f]] # Input for the SVM

		print('Original number of features :', len(X_train.columns))
		print("Number of features after elimination: %f [Acc: %.4f]" %(rfecv.n_features_, rfecv.grid_scores_[rfecv.n_features_])) 
		#print("Features ID:","\n", best_f )
		#print("Grid score:", rfecv.grid_scores_)#Score of the estimator produced when is trained with the i-th subset of features
		
		fit_SVC(X_train_select, y_train, X_test_select, y_test, best_f)
		
		k_iteration += 1


def main(args):
	
	X = pd.read_csv("Data/Train_call.txt", delimiter="\t")
	
	labels = pd.read_csv("Data/Train_clinical.txt",  delimiter="\t")
	do_RFECV(X, labels)
	
	return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
