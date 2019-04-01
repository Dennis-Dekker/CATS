# Author: Chao (Cico) Zhang
# Date: 31 Mar 2017
# Usage: Rscript run_model.R -i unlabelled_sample.txt -m model.pkl -o output.txt
# If you are using python, please use the Python script template instead.
# Set up R error handling to go to stderr
options(show.error.messages=F, error=function(){cat(geterrmessage(),file=stderr());q("no",1,F)})

# Import required libraries
# You might need to load other packages here.
suppressPackageStartupMessages({
  library('getopt')
  library('caret')
})

# Take in trailing command line arguments
args <- commandArgs(trailingOnly = TRUE)

# Get options using the spec as defined by the enclosed list
# Read the options from the default: commandArgs(TRUE)
option_specification <- matrix(c(
  'input', 'i', 2, 'character',
  'model', 'm', 2, 'character',
  'output', 'o', 2, 'character'
), byrow=TRUE, ncol=4);

# Parse options
options <- getopt(option_specification);

# Start your coding

# suggested steps
# Step 1: load the model from the model file (options$model)
# Step 2: apply the model to the input file (options$inout) to do the prediction
# Step 3: write the prediction into the desinated output file (options$output)

# End your coding
message ("Done!")
