import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import TruncatedSVD, PCA, KernelPCA, SparsePCA
from sklearn.feature_selection import RFECV
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV,StratifiedKFold,KFold, ParameterGrid
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn import preprocessing
import matplotlib.pyplot as plt
import sys, time, threading



### Load data
df_data_call = pd.read_csv("Data/Train_call.txt", delimiter="\t")
df_x = df_data_call.transpose()
X_data = df_x.iloc[4:,]


df_data_clinical = pd.read_csv("Data/Train_clinical.txt",  delimiter="\t")
y = df_data_clinical["Subgroup"]
#to df
df_y= pd.DataFrame(y)

##Preprocess
#Change categorical to numerical(classes)
le = preprocessing.LabelEncoder()
le.fit(np.unique(df_y.values))
y = df_y.apply(le.transform)
y_data= y.values.ravel()

##Hyperparameters estimation for SVC
# Split the dataset, 80% Train
X_train, X_test, y_train, y_test = train_test_split(
X_data, y_data, test_size=0.2, random_state=0)
    
# Set the parameters to test 
tunned_parameters = [{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]    

#Cross validation parameters 
cv_parameters =KFold(n_splits=7, shuffle=True, random_state=0)

#Hyperparameters tunning using GridSearch
GSCV=GridSearchCV(estimator=SVC(), param_grid=tunned_parameters, cv=cv_parameters, verbose=2,n_jobs=-1)

#train a model with each set of parameters
GSCV.fit(X_train, y_train)
best_hyperparameters= GSCV.best_params_
    
##Pass the hyperparameters
svc =SVC(**best_hyperparameters)    

##Recursive Feature Elimination with Cross Validation (RFECV)
#StratifiedKFold(5) this is similar to Kfold with the difference that the fold are generated in a way that keeps the percentage of each class.
rfecv = RFECV(estimator=svc, step=1,min_features_to_select=5, cv=StratifiedKFold(5),scoring='accuracy',n_jobs=-1)
#fit the data
rfecv.fit(X_data, y_data)#Takes some time to do the fit.

#Best features
best_features = rfecv.get_support(1)
X = X_data[X_data.columns[best_features]] # Input for the SVM
print("###############CATS Winners#####################################")
print('Original number of features :', len(X_data.columns))
print("Final number of features with CV:",rfecv.n_features_) 
print("Features name:","\n", best_features)
print("Grid score:", rfecv.grid_scores_)#Score of the estimator produced when is trained with the i-th subset of features
print("################################################################")


##################################################################################################
##Plot PCA with using the best features(Im curious)
pca = PCA(n_components=2)#run PCA wtih 2 components
principalComponents = pca.fit_transform(X)

##Edition to plot adding color by class
df_full = X.rename_axis(['Sample']).reset_index()
df_test=pd.merge(df_full, df_data_clinical, on ="Sample")#add classes
  
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
             
df_PC_Class = pd.concat([principalDf, df_test["Subgroup"]],axis=1)
df_PC_Class.columns=["PC1","PC2","Subgroup"]

#plot by class
pca_color=sns.pairplot(x_vars=["PC1"], y_vars=["PC2"], data=df_PC_Class, hue="Subgroup", height=5)
pca_color.set(xlabel = "PC1 (" + str(round(pca.explained_variance_ratio_[0]*100, 1)) + "%)")
pca_color.set(ylabel = "PC2 (" + str(round(pca.explained_variance_ratio_[1]*100, 1)) + "%)")
plt.show()
    
