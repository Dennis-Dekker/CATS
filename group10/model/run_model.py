#!/usr/bin/env python3
"""Reproduce your result by your saved model.

This is a script that helps reproduce your prediction results using your saved
model. This script is unfinished and you need to fill in to make this script
work. If you are using R, please use the R script template instead.

The script needs to work by typing the following commandline (file names can be
different):

python3 run_model.py -i unlabelled_sample.txt -m model.pkl -o output.txt

"""

# author: Chao (Cico) Zhang
# modified by: Dennis Dekker
# date: 31 Mar 2017

import argparse
import sys

# Start your coding

# import the library you need here
import pickle
import csv
import pandas as pd

# Needed to open the .pkl file
class Model:
    def __init__(self, svm, c_l1, best_features, accuracy_validate, best_params, n_features):
        self.svm = svm
        self.c_l1 = c_l1
        self.best_features = best_features
        self.accuracy_validate = accuracy_validate
        self.best_params = best_params
        self.n_features = n_features


# End your coding


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Reproduce the prediction')
    parser.add_argument('-i', '--input', required=True, dest='input_file',
                        metavar='unlabelled_sample.txt', type=str,
                        help='Path of the input file')
    parser.add_argument('-m', '--model', required=True, dest='model_file',
                        metavar='model.pkl', type=str,
                        help='Path of the model file')
    parser.add_argument('-o', '--output', required=True,
                        dest='output_file', metavar='output.txt', type=str,
                        help='Path of the output file')
    # Parse options
    args = parser.parse_args()

    if args.input_file is None:
        sys.exit('Input is missing!')

    if args.model_file is None:
        sys.exit('Model file is missing!')

    if args.output_file is None:
        sys.exit('Output is not designated!')

    # Start your coding

    # suggested steps
    # Step 1: load the model from the model file
    # Step 2: apply the model to the input file to do the prediction
    # Step 3: write the prediction into the desinated output file

    # Load model and test data
    with open(args.model_file, 'rb') as model_input:
        # Needs Model class to open!
        model = pickle.load(model_input)
    test = pd.read_csv(args.input_file, sep="\t")

    # Remove unneeded columns
    test = test.drop(["Chromosome", "Start", "End", "Nclone"], axis=1).transpose()

    # Extract only best features from test data
    x_test = model.best_features.transform(test)

    # Predict
    prediction = pd.DataFrame(model.svm.predict(x_test), columns=["Subgroup"])
    samples = pd.DataFrame(test.index, columns=["Sample"])
    final_predictions = samples.join(prediction)

    # Save predictions to .txt file
    final_predictions.to_csv(args.output_file, header=True, index=None, sep='\t', quoting=csv.QUOTE_ALL)

    # End your coding


if __name__ == '__main__':
    main()
