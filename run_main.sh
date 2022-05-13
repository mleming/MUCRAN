#!/bin/bash

# The three essential arguments for this are the label to classify by, the 
# confounds to regress, and the var_file that stores the paths to all of the
# data as well as the labels themselves.

python3 main.py --var_file ../../pandas/cache/all_vars_2.pkl --l AlzStage -c SexDSC RepetitionTime
