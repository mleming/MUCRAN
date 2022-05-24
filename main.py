#!/usr/bin/python

# Code for MUCRAN (Multi-confound regression adversarial network) for 3D image 
# classification and regression. Combined with data matching methods.
# Designed for use on MGH MRI data, but could be adapted for
# any data that is in a pandas dataframe.
# Author: Matt Leming
# Requires an input Pandas dataframe with indices being the .npy files to read
# in and associated columns to be the confounds and labels associated with each
# datapoint. For instance:
# 
# python main.py -f vars.pkl -l Alz_Label -c SexDSC RepetitionTime Age
#
# Dataframes are verified at the start of the task to have columns with the
# appropriate label and confounds in them.

import os,sys,glob,json,argparse,time,datetime,datefinder
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from dateutil import relativedelta,parser
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from models import *
matplotlib.use('Agg') # For outputting figures on clusters
from tensorflow.keras import backend as K
from utils import *
args = get_args()
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
from sklearn import preprocessing
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.optimizers import *
from tensorflow.keras.regularizers import *
from sklearn.metrics import roc_curve, auc
from batch_scheduler import BatchScheduler
from utils import *
all_vars = pd.read_pickle(args.var_file)
validate_all_vars(all_vars,args)

model_dir = os.path.join(args.working_dir,'models')
if not os.path.isdir(model_dir): os.makedirs(model_dir)

# Can use these variables to divide between test and train set based on
# confound values, or to only use a specific part of the data overall.
# Examples are commented
train_val_ranges = {}
test_val_ranges = {}

#train_val_ranges["Modality"] = ["MR"]
#test_val_ranges["Modality"]  = ["MR"]
#
#train_val_ranges['ProtocolNameSimplified'] = ["T1_AX"]
#test_val_ranges['ProtocolNameSimplified']  = ["T1_AX"]
#
#train_val_ranges['SexDSC'] = "MALE"
#test_val_ranges['SexDSC'] = "FEMALE"

dataloader = BatchScheduler(args,all_vars,train_val_ranges)

encoder, input_img , feature_dense = get_encoder(dataloader.in_dim,args)
regressor = get_regressor(dataloader.out_dim,feature_dense,args)
# The regressor uses SGD while the encoder uses an Adam optimizer		
regressor.compile(optimizer=SGD(lr=args.lr),
	loss="binary_crossentropy")
regressor.trainable=False
mucran = Model(inputs=input_img,outputs=regressor(feature_dense))
mucran.compile(optimizer=Adam(lr=args.lr),
	loss="binary_crossentropy")
if args.verbose:
	print("\n\nMUCRAN (Combined Model)\n")
	mucran.summary()
	print("\n\nREGRESSOR\n")
	regressor.summary()

print(len(dataloader))
for i,(X,Y,YC,YC_dud) in enumerate(dataloader):
	print("Iter %d / %d" % (i,len(dataloader)))
	for _ in range(args.iters_per_batch):
		mucran.train_on_batch(X,YC_dud)
		regressor.train_on_batch(encoder.predict(X),YC)

## Outputs the test set evaluations
output_test(args,test_val_ranges,os.path.join(model_dir,"test_results.json"),
	os.path.join(model_dir,"test_predictions.json"),mucran,all_vars)
