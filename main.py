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
matplotlib.use('Agg') # For outputting figures on clusters
from tensorflow.keras import backend as K

parser = argparse.ArgumentParser(description = "Used for testing models")
parser.add_argument('-c','--confounds',nargs="+",required=True,
	help="List of confounds to regress")
parser.add_argument('-l','--label',type=str,required=True,
	help="Label to train")
parser.add_argument('-f','--var_file',type=str,required=True)

parser.add_argument('--train_confounded',default=False,action='store_true',
	help="Trains a confounded model rather than a regressed model")
parser.add_argument('--working_dir',default=\
	os.path.dirname(os.path.realpath(__file__)),
	help="Directory in which to save everything")
parser.add_argument('--match_confounds',default=[],nargs='+',
	help="Arguments to match for, but not necessarily regress")
parser.add_argument('--train_label_only',default=False,action='store_true',
	help="Trains a label-only model rather than a regressed model")
parser.add_argument('--gpu',default="0")
parser.add_argument('--num_iters',default=100,type=int)
parser.add_argument('--load_only',action='store_true',default=False)
parser.add_argument('--meta',type=str,default="")
parser.add_argument('--no_train_if_exists',default=False,action='store_true')
parser.add_argument('--get_all_test_set',action='store_true',default=False)
parser.add_argument('--test_predictions_filename',
	default='test_predictions.json',type=str)
parser.add_argument('--total_size_limit',default=None,type=int)
parser.add_argument('--no_train_confounds',default=False,action='store_true')
parser.add_argument('--nobatch',default=False,action='store_true')
parser.add_argument('--number',type=int,default=0)
parser.add_argument('--balance_size',type=int,default=33000,
	help="Total amount of data per label to be loaded into main memory")
parser.add_argument('--nomatch',action='store_true',default=False,
	help="Data matching is not performed")
parser.add_argument('--threetrain',action='store_true',default=False,
	help="Flag to train confounded, regressed, and label-only models at once.")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
from sklearn import preprocessing
from supporting_functions import *
from tensorflow.keras.layers import *
#from get_3D_models import *
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import *
from tensorflow.keras.regularizers import *
from sklearn.metrics import roc_curve, auc

def YC_conv(Y,C,y_weight):
	Y = np.reshape(Y,(Y.shape[0],1,Y.shape[1]))
	Y_ = Y
	for j in range(y_weight-1):
		Y_ = np.concatenate((Y_,Y),axis=1)
	Y = Y_
	Y = np.concatenate((Y,np.zeros((Y.shape[0],
		Y.shape[1],C.shape[2]-Y.shape[2]))),axis=2)
	YC = np.concatenate((Y,C),axis=1)
	C_dud = np.zeros(C.shape)
	C_dud[:,:,0] = 1
	YC_dud = np.concatenate((Y,C_dud),axis=1)
	return YC,YC_dud

# Legacy code from the previous version of this script. Will likely use this in
# the future to save models.

def parsedate(d,date_format="%Y-%m-%d %H:%M:%S"):
	for match in datefinder.find_dates(d.replace("_"," ")): return match
	return datetime.datetime.strptime(d.split(".")[0],date_format)

def get_title_str(args,train_label_only,train_confounded):
	title_str = "%s_conf_gan" % (args.label)
	if args.no_train_confounds:
		title_str += "_no_confound_reg"
	if train_confounded:
		title_str += "_converg"
	if train_label_only:
		title_str += "_dud"
	if args.nobatch:
		title_str += "_nobatch"
	if args.nomatch:
		title_str += "_nomatch"
	if True:
		title_str += "_%d" % args.number
	return title_str

assert(not(args.train_label_only and args.train_confounded))

# Validate the input pandas file
assert(os.path.isfile(args.var_file))
assert(os.path.splitext(args.var_file)[1] == ".pkl")
all_vars = pd.read_pickle(args.var_file)

for c in args.confounds:
	if c not in all_vars.columns:
		raise Exception("Confound %s not in columns of %s"%(c,args.var_file))

if args.label not in all_vars.columns:
	raise Exception("Label %s not in columns of %s"%(args.label,args.var_file))

for index in all_vars.index:
	if os.path.splitext(index)[1] != ".npy":
		raise Exception(("Indices of %s must all be .npy files: "+\
			"exception at index %s") % (args.var_file,index))
#	if not os.path.isfile(index):
#		raise Exception("Index of %s not a file: %s" % (args.var_file,index))

# Trains regressed, confounded, and label-only models simultaneously, for later
# comparison
if args.threetrain:
	title_strs = [get_title_str(args,False,False),
		get_title_str(args,False,True),get_title_str(args,True,False)]
else:
	title_strs = [get_title_str(args,args.train_label_only,args.train_confounded)]

label = args.label

models_dir = os.path.join(args.working_dir,'models')

if not os.path.isdir(models_dir):
	os.makedirs(models_dir)

path_files = {}
for t in title_strs:
	title_str = t
	path_files[t] = {}
	current_model_dir = os.path.join(models_dir,title_str)
	
	if args.meta != "":
		current_model_dir = os.path.join(models_dir,
					 args.label,"_%s_%s" % \
					 (title_str,args.meta))
	if not os.path.isdir(current_model_dir):
		os.makedirs(current_model_dir)
	path_files[t]["current_model_dir"] = current_model_dir
	path_files[t]["best_model_dir"] = os.path.join(current_model_dir,'model')
	path_files[t]["test_predictions_file"] = os.path.join(current_model_dir,
		'test_predictions.json')
	path_files[t]["best_model_state"] = os.path.join(current_model_dir,
		'state.json')
	path_files[t]["parameters_state"] = os.path.join(current_model_dir,
		'parameters.json')
	path_files[t]["np_dir"] = os.path.join(current_model_dir,'npy')
	path_files[t]["output_covars_savepath"] = os.path.join(current_model_dir,
					'cache','%s_balanced.pkl'%args.label)
	path_files[t]["output_selection_savepath"] = os.path.join(current_model_dir,
					'cache','%s_balanced.npy'%args.label)
	path_files[t]["output_test_predictions"] = os.path.join(current_model_dir,
					args.test_predictions_filename)
	path_files[t]["output_results"] = os.path.join(current_model_dir,
		'test_results.json')
	path_files[t]["output_grad_sample"] = os.path.join(current_model_dir,
		'grad_samples')
	if not os.path.isdir(path_files[t]["output_grad_sample"]):
		os.makedirs(path_files[t]["output_grad_sample"])
	path_files[t]["output_regressor_loss"] = os.path.join(current_model_dir,
		'regressor_loss.png')
	path_files[t]["output_encoder_loss"] = os.path.join(current_model_dir,
		'encoder_loss.png')

imsize = (96,96,96)

# Confounds from all_vars that are balanced by using dataset matching
# (Leming et al 2021)

if args.nomatch:
	args.match_confounds = []

subcycle = [args.label]
balance_cycle=[args.label]

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
y_weight = 6

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

args.confounds = sorted(args.confounds)

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
		YC,YC_dud = YC_conv(Y,C,y_weight)
		file_list_dict[b] = file_list_dict[b][batch_size:]
		file_list_dict[c] = file_list_dict[c][batch_size:]
		assert(X.shape[0] > 0)
		
		# Initializes the encoder
		if encoder is None:
			input_img = Input(shape = (imsize[0],imsize[1],imsize[2],1))
			latent_dim=512
			ft_bank_baseline = 128
			n_intermed_features = 1024
		
			feature = Conv3D(ft_bank_baseline, kernel_size=3,
			padding='same',strides=2)(input_img)
			feature = LeakyReLU(alpha=0.3)(feature)
			feature = BatchNormalization()(feature)
	
			feature = Conv3D(ft_bank_baseline*2, kernel_size=3,
				padding='same',strides=2)(feature)
			feature = LeakyReLU(alpha=0.3)(feature)
			feature = BatchNormalization()(feature)
			
			feature = Conv3D(ft_bank_baseline*2, kernel_size=3,
				padding='same',strides=2)(feature)
			feature = LeakyReLU(alpha=0.3)(feature)
			feature = BatchNormalization()(feature)
				
			feature = Conv3D(ft_bank_baseline*2, kernel_size=3,
				padding='same',strides=2)(feature)
			feature = LeakyReLU(alpha=0.3)(feature)
			feature = BatchNormalization()(feature)
		
			feature = Flatten()(feature)
			feature = Dense(latent_dim*4)(feature)
			feature = LeakyReLU(alpha=0.3)(feature)
			feature = BatchNormalization()(feature)
			
			feature = Dense(latent_dim*4)(feature)
			feature = LeakyReLU(alpha=0.3)(feature)
			feature = BatchNormalization()(feature)
			
			feature = Dense(n_intermed_features,
				kernel_regularizer=l2(1e-4))(feature)
			
			feature_dense = Flatten()(feature)
			encoder = Model(inputs=input_img,outputs=feature_dense)
			inputs_x = Input(shape=(n_intermed_features,))
			feature = Dense(latent_dim*4,kernel_regularizer=l2(1e-4))(inputs_x)
			feature = LeakyReLU(alpha=0.3)(feature)
			if args.nobatch:
				feature = Dropout(0.3)(feature)
			else:
				feature = BatchNormalization()(feature)
			feature = Dense(latent_dim*2)(feature)
			feature = LeakyReLU(alpha=0.3)(feature)
			if args.nobatch:
				feature = Dropout(0.3)(feature)
			else:
				feature = BatchNormalization()(feature)
			feature = Dense(latent_dim*2,kernel_regularizer=l2(1e-4))(feature)
			feature = LeakyReLU(alpha=0.3)(feature)
			cf = Concatenate(axis=1)([Reshape((1,YC.shape[2]))\
				(Dense(YC.shape[2],activation='softmax',kernel_regularizer=l2(1e-4))(feature))\
				 for _ in range(YC.shape[1])])

			# The regressor uses SGD while the encoder uses an Adam optimizer		
			regressor = Model(inputs = inputs_x,outputs=cf)
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
				path_files[title_str]["mucran"] = mucran
				path_files[title_str]["regressor"] = regressor
				path_files[title_str]["encoder"]   = encoder
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
				YC1_losses[0] = np.mean(YC_losses[:y_weight])
				YC1_losses[1:] = YC_losses[y_weight:]
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
				plt.savefig(path_files[t]["output_regressor_loss"])
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
				YC_dud1_losses[0] = np.mean(YC_dud_losses[:y_weight])
				YC_dud1_losses[1:] = YC_dud_losses[y_weight:]
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
				plt.savefig(path_files[t]["output_encoder_loss"])
				plt.clf()
	
	for mucran,title_str in zip(mucrans,title_strs):
		mucran.save(path_files[title_str]["best_model_dir"])
	time1 = (time.time() - time1)/3600
	all_times.append(time1)
	print("%.4f est. hours remaining" % \
		 (np.mean(all_times) * ((int(args.balance_size/(batch_size))-i + 1))))
	if skip: break


## Outputs the test set evaluations

def output_test(test_val_ranges,output_results,test_predictions_file,mucran,
	X_files = None,return_Xfiles = False):
	pred   = None
	c_pred = None
	Y      = None
	C      = None
	cert   = None
	results = {}
	np.random.seed(args.number)
	for b in balance_cycle:
		if len(balance_cycle) > 1:
			test_predictions_file = test_predictions_file.replace(".json",
				"_%s.json"%b)
		if X_files is None:
			[X_files],_ = get_balanced_filename_list(b,match_confounds,
				selection_ratios=[1],
				total_size_limit=balance_size,
				non_confound_value_ranges = test_val_ranges,
				all_vars=all_vars)
		temp = X_files
		while len(X_files) > 0:
			X_,Y_,C_ = get_data_from_filenames(X_files[:batch_size],
				test_var,confounds=confounds,
				return_as_strs = False,
				unique_test_vals = None,all_vars=all_vars)
			if test_var == "ICD":
				Y_ = Y_[:,selection]
			#YC_,YC_dud_ = YC_conv(Y_,C_,y_weight)
			YC_pred = mucran.predict(X_)
			pred_ = np.mean(YC_pred[:,:y_weight,:],axis=1)
			c_pred_ = YC_pred[:,y_weight:,:]
			cert_ = None
			####
			if Y is None:
				X = X_
				Y = Y_
				C = C_
				pred = pred_
				c_pred = c_pred_
				cert = cert_
				#X,Y,C,pred,c_pred,cert = X_,Y_,C_,pred_,c_pred_,cert_
			else:
				pred   = np.concatenate((pred,pred_),     axis=0)
				c_pred = np.concatenate((c_pred,c_pred_), axis=0)
				Y      = np.concatenate((Y,Y_),           axis=0)
				C      = np.concatenate((C,C_),           axis=0)
				#cert   = np.concatenate((cert,cert_),     axis=0)
			X_files = X_files[batch_size:]
		X_files = temp
		save_dict = {}
		#print("cert.shape: %s" % str(cert.shape))
		for i in range(Y.shape[0]):
			X_file = X_files[i]
			save_dict[X_file] = [[float(_) for _ in pred[i,:]],
				[float(_) for _ in Y[i,:]]]#,float(cert[i])]
		json.dump(save_dict,open(test_predictions_file,'w'),indent=4)
		pred_bin = np.zeros(Y.shape)
		Y_bin = np.zeros(Y.shape)
		for i in range(pred_bin.shape[0]):
			pred_bin[i,np.argmax(pred[i,:Y.shape[1]])] = 1
			Y_bin[i,np.argmax(Y[i,:])] = 1
		roc_aucs = []
		print("Y AUROCS")
		results[args.test_variable] = {}
		print("Y.shape: %s" % str(Y.shape))
		print("pred_bin.shape: %s" % str(pred_bin.shape))
		results[args.test_variable]["Y_acc"] = \
			float(np.mean(np.all(Y == pred_bin,axis=1)))
		for i in range(Y.shape[1]):
			fpr, tpr, threshold = roc_curve(Y[:,i],pred[:,i])
			roc_auc = auc(fpr, tpr)
			roc_aucs.append(roc_auc)
			print("%s AUROC: %s" % (output_labels[i],str(roc_auc)))
			results[args.test_variable][output_labels[i]] = float(roc_auc)
		results[args.test_variable]["Mean AUROC"] = float(np.mean(roc_aucs))
		
		print("Mean AUROC: % s" % str(np.mean(roc_aucs)))
		print("Y acc: %f" % results[args.test_variable]["Y_acc"])
		#print("Y AUROC: %s" % str(roc_auc))

		print("+++")
		print("MAX CONFOUND AUROCS")
		for i in range(len(confounds)):
			confound = confounds[i]
			roc_aucs = []
			roc_aucs_counts = []
			for j in range(C.shape[2]):
				if np.any(C[:,i,j] == 1):
					fpr, tpr, threshold = roc_curve(C[:,i,j],c_pred[:,i,j])
					roc_auc = auc(fpr, tpr)
					if not iz_nan(roc_auc):
						#roc_aucs[roc_auc] = int(np.sum(C[:,i,j]))
						roc_aucs.append(roc_auc)
						roc_aucs_counts.append(int(np.sum(C[:,i,j])))
			weighted_mean = np.sum([c1*c2 for c1,c2 in zip(roc_aucs,roc_aucs_counts)]) /\
					 np.sum(roc_aucs_counts)
			try:
				results[confound] = {}
				if len(roc_aucs) == 0:
					print("No AUCs for %s" % confound)
				else:
					mroc = int(np.argmax(roc_aucs))
					meanroc = np.mean(roc_aucs)
					print(("%s: %f (max); %d (num in max) ;"+\
						" %f (mean); %f (weighted mean)") \
						% (confound,roc_aucs_counts[mroc],roc_aucs[mroc],meanroc,
						weighted_mean))
					results[confound]["MAX AUROC"] = float(roc_aucs[mroc])
					results[confound]["NUM IN MAX"] = float(roc_aucs_counts[mroc])
					results[confound]["MEAN AUROC"] = float(meanroc)
					results[confound]["WEIGHTED MEAN"] = float(weighted_mean)
			except:
				print("Error in outputting %s" % confound)

	json.dump(results,open(output_results,'w'),indent=4)
	if return_Xfiles: return X_files

xfilelists = None
for mucran,t in zip(mucrans,title_strs):
	current_model_dir = path_files[t]["current_model_dir"]
	xfilelists = output_test(test_val_ranges,output_results,
		args.test_predictions_file,mucran,
		return_Xfiles = True,X_files=xfilelists)
