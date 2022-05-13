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
from supporting_functions import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.optimizers import *
from tensorflow.keras.regularizers import *
from sklearn.metrics import roc_curve, auc

all_vars = pd.read_pickle(args.var_file)
validate_all_vars(all_vars,args)

# Trains regressed, confounded, and label-only models simultaneously, for later
# comparison

if args.threetrain:
	title_strs = [get_title_str(args,False,False),
		get_title_str(args,False,True),get_title_str(args,True,False)]
else:
	title_strs = [get_title_str(args,args.train_label_only,args.train_confounded)]

models_dir = os.path.join(args.working_dir,'models')
if not os.path.isdir(models_dir): os.makedirs(models_dir)
paths_dict = get_paths_dict(args,title_strs,models_dir)

# Confounds from all_vars that are balanced by using dataset matching
# (Leming et al 2021)

if args.nomatch:
	args.match_confounds = []

subcycle     = [args.label]
balance_cycle= [args.label]

assert(np.all([s in balance_cycle for s in subcycle]))
assert(len(subcycle) == len(list(set(subcycle))))

selection = np.zeros((len(balance_cycle,)),dtype=bool)
for i in range(selection.shape[0]):
	if balance_cycle[i] in subcycle or len(subcycle) == 0:
		selection[i] = True

if len(subcycle) > 0:
	balance_cycle = subcycle

assert(np.sum(selection) == len(subcycle) or \
	(len(subcycle) == 0 and np.sum(selection) == len(balance_cycle)))

# The number of times the main signal is repeated in the output. Offers more
# weight to that particular output compared to confounds.

# Number of data files loaded into memory at a time
minibatch_size=24
batch_size = 48
epochs_per_batch = 5

grad_n = int(args.balance_size/(batch_size*10))

randomize_order = True

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

file_list_dict = {}
confound_file_list_dist = {}
skip = False
i_c = -1

label_and_confound = [args.label] + args.confounds

grad_samples = None
encoder = None
encoder_losseses = None
regressor_losseses = None

all_times = []

# Loops through data and balances iteratively by certain confounds.
for i in range(int(args.balance_size/(batch_size))):
	time1 = time.time()
	for b in balance_cycle:
		print("(%d / %d): %s" % (i,int(args.balance_size/batch_size),b))
		i_c = (i_c + 1) % len(args.confounds)
		c = args.confounds[i_c]
		if b not in file_list_dict or len(file_list_dict[b]) == 0:
			[X_files],_ = get_balanced_filename_list(b, args.match_confounds,
				selection_ratios=[1], total_size_limit=args.balance_size,
				non_confound_value_ranges = train_val_ranges,verbose=False,
				all_vars=all_vars)
			print("Num files in %s: %d" % (b,len(X_files)))
			if randomize_order:
				random.shuffle(X_files)
			file_list_dict[b] = X_files
		if c not in file_list_dict or len(file_list_dict[c]) == 0:
			[X_files],_ = get_balanced_filename_list(c,args.match_confounds,
				selection_ratios=[1],total_size_limit=args.balance_size,
				non_confound_value_ranges = train_val_ranges,verbose=False,
				all_vars=all_vars)
			if len(X_files) < batch_size:
				print(("Not enough samples found for %s "+\
					"-- getting random ones") % c)
				[X_files],_=get_balanced_filename_list("SexDSC",
					args.match_confounds,
					selection_ratios=[1], total_size_limit=args.balance_size,
				    non_confound_value_ranges=train_val_ranges, verbose=False,
					all_vars=all_vars)
			if randomize_order:
				random.shuffle(X_files)
			file_list_dict[c] = X_files
		X,Y,C,choice_arr,output_label_and_confound = get_data_from_filenames(\
				np.concatenate((file_list_dict[b][0:batch_size],
				file_list_dict[c][0:batch_size]),axis=0),
				args.label,confounds=args.confounds,
				return_as_strs = False,unique_test_vals = None,
				all_vars=all_vars,return_choice_arr=True,return_as_dict=False,
				dict_obj=None)

		if grad_samples is None: grad_samples = X[:4,:,:,:]
		YC,YC_dud = YC_conv(Y,C,args.y_weight)
		file_list_dict[b] = file_list_dict[b][batch_size:]
		file_list_dict[c] = file_list_dict[c][batch_size:]
		assert(X.shape[0] > 0)
		
		# Initializes the model. Models are initialized only after the output
		# size is determined.
		if encoder is None:
			encoder, input_img , feature_dense = get_encoder(args.imsize,args)
			regressor = get_regressor(YC.shape,feature_dense,args)
			# The regressor uses SGD while the encoder uses an Adam optimizer		
			regressor.compile(optimizer=SGD(lr=0.0002),
				loss="binary_crossentropy")
			regressor.trainable=False
			mucran = Model(inputs=input_img,outputs=regressor(feature_dense))
			mucran.compile(optimizer=Adam(lr=0.0002),
				loss="binary_crossentropy")
			print("\n\nMUCRAN (Combined Model)\n")
			mucran.summary()
			print("\n\nREGRESSOR\n")
			regressor.summary()
			regressors = [regressor]
			encoders   = [encoder]
			mucrans = [mucran]

			if args.threetrain:
				for _ in range(2):
					regressor_copy = keras.models.clone_model(regressor)
					regressor_copy.compile(optimizer=SGD(lr=0.0002),loss="binary_crossentropy")
					regressor_copy.trainable = False
					encoder_copy = keras.models.clone_model(encoder)
					encoders.append(encoder_copy)
					mucran_copy = Model(inputs=encoder_copy.input,outputs=regressor_copy(encoder_copy.outputs))
					mucran_copy.compile(optimizer=Adam(lr=0.0002), loss="binary_crossentropy")
					regressors.append(regressor_copy)
					mucrans.append(mucran_copy)
			for mucran,regressor,encoder,title_str in zip(mucrans,regressors,encoders,title_strs):
				paths_dict[title_str]["mucran"] = mucran
				paths_dict[title_str]["regressor"] = regressor
				paths_dict[title_str]["encoder"]   = encoder
		print(i)
		for j in range(epochs_per_batch):
			perm = np.random.permutation(X.shape[0])
			for k in range(int(X.shape[0]/minibatch_size)-1):
				jj = perm[k*minibatch_size:(k+1)*minibatch_size]
				if args.threetrain:
					mucrans[0].train_on_batch(X[jj],YC_dud[jj])
					regressors[0].train_on_batch(encoders[0].predict(X[jj]),YC[jj])
					mucrans[1].train_on_batch(X[jj],YC[jj])
					regressors[1].train_on_batch(encoders[1].predict(X[jj]),YC[jj])
					mucrans[2].train_on_batch(X[jj],YC_dud[jj])
					regressors[2].train_on_batch(encoders[2].predict(X[jj]),YC_dud[jj])
				else:
					if args.train_confounded:
						mucran.train_on_batch(X[jj],YC[jj])
					else:
						mucran.train_on_batch(X[jj],YC_dud[jj])
					if args.train_label_only:
						regressor.train_on_batch(encoder.predict(X[jj]),YC_dud[jj])
					else:
						regressor.train_on_batch(encoder.predict(X[jj]),YC[jj])
			if regressor_losseses is None:
				regressor_losseses = [None for _ in range(len(mucrans))]
				encoder_losseses = [None for _ in range(len(mucrans))]
			for jj in range(len(mucrans)):
				mucran = mucrans[jj]
				regressor_losses = regressor_losseses[jj]
				encoder_losses = encoder_losseses[jj]
				t = title_strs[jj]
				Y_preds = mucran.predict(X)
				YC_losses = np.mean(
					np.squeeze(
					np.mean(
						K.binary_crossentropy(YC[:,:,:].astype('float64'),
						Y_preds[:,:,:].astype('float64')),
						axis=0)),
					axis=1)
				YC_losses = np.reshape(YC_losses,(len(YC_losses),1))
				YC1_losses = np.zeros((len(args.confounds)+1,1))
				YC1_losses[0] = np.mean(YC_losses[:args.y_weight])
				YC1_losses[1:] = YC_losses[args.y_weight:]
				YC_losses = YC1_losses
				if regressor_losses is None:
					regressor_losses = YC_losses
				else:
					regressor_losses = np.concatenate((regressor_losses,YC_losses),
						axis=1)
				regressor_losseses[jj] = regressor_losses
			
				for kk in range(len(label_and_confound)-1,-1,-1):
					if label_and_confound[kk] == args.label:
						plt.plot(np.arange(regressor_losses.shape[1]),
							regressor_losses[kk,:],label=label_and_confound[kk],linewidth=3,
							color='black')
					else:
						plt.plot(np.arange(regressor_losses.shape[1]),
							regressor_losses[kk,:],label=label_and_confound[kk])
				plt.title("Regressor")
				plt.xlabel("Iteration")
				plt.ylabel("Loss")
				plt.legend()
				plt.savefig(paths_dict[t]["output_regressor_loss"])
				plt.clf()
	
				YC_dud_losses = np.mean(
					np.squeeze(
					np.mean(
						K.binary_crossentropy(YC_dud[:,:,:].astype('float64'),
						Y_preds[:,:,:].astype('float64')),
						axis=0)),
					axis=1)
				YC_dud_losses = np.reshape(YC_dud_losses,(len(YC_dud_losses),1))
				YC_dud1_losses = np.zeros((len(args.confounds)+1,1))
				YC_dud1_losses[0] = np.mean(YC_dud_losses[:args.y_weight])
				YC_dud1_losses[1:] = YC_dud_losses[args.y_weight:]
				YC_dud_losses = YC_dud1_losses
				if encoder_losses is None:
					encoder_losses = YC_dud_losses
				else:
					encoder_losses = np.concatenate((encoder_losses,YC_dud_losses),
						axis=1)
				encoder_losseses[jj] = encoder_losses
				for kk in range(len(label_and_confound)-1,-1,-1):
					if label_and_confound[kk] == args.label:
						plt.plot(np.arange(encoder_losses.shape[1]),
							encoder_losses[kk,:],label=label_and_confound[kk],linewidth=3,
							color='black')
					else:
						plt.plot(np.arange(encoder_losses.shape[1]),
							encoder_losses[kk,:],label=label_and_confound[kk])
				plt.xlabel("Iteration")
				plt.ylabel("Loss")
				#plt.title("%s encoder loss" % args.label)
				plt.title("Encoder")
				plt.legend()
				plt.savefig(paths_dict[t]["output_encoder_loss"])
				plt.clf()
	
	for mucran,title_str in zip(mucrans,title_strs):
		mucran.save(paths_dict[title_str]["best_model_dir"])
	time1 = (time.time() - time1)/3600
	all_times.append(time1)
	print("%.4f est. hours remaining" % \
		 (np.mean(all_times) * ((int(args.balance_size/(batch_size))-i + 1))))
	if skip: break

## Outputs the test set evaluations
xfilelists = None
for mucran,t in zip(mucrans,title_strs):
	current_model_dir = paths_dict[t]["current_model_dir"]
	xfilelists = output_test(test_val_ranges,output_results,
		args.test_predictions_file,mucran,
		return_Xfiles = True,X_files=xfilelists)
