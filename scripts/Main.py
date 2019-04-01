#!/usr/bin/env python3

import argparse
import sys
import pandas as pd
from sklearn.decomposition import PCA
import seaborn as sns


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
    print(finalDf.iloc[0:5,0:5])
    
    pca_color=sns.pairplot(x_vars=["principal component 1"], y_vars=["principal component 2"], data=finalDf, hue="Subgroup", height=5)
    path_PCA_figure_color = "../images/PCA_color.png"
    pca_color.set(xlabel = "PC1 (" + str(round(pca.explained_variance_ratio_[0]*100, 1)) + "%)")
    pca_color.set(ylabel = "PC2 (" + str(round(pca.explained_variance_ratio_[1]*100, 1)) + "%)")
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
    print(data.iloc[0:5,0:5])
    
    # Simple pca
    calculate_pca(data, labels)

if __name__ == '__main__':
    main()
