import os,argparse
import numpy as np

def get_args():
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
	parser.add_argument('--imsize',default=(96,96,96),help='Size of the input image')
	parser.add_argument('--y_weight',default=6,type=int,help='The number of '\
		+'times the main signal is repeated in the output. Offers more weight'+\
		' to that particular output compared to confounds.')
	args = parser.parse_args()
	args.confounds = sorted(args.confounds)
	assert(not(args.train_label_only and args.train_confounded))
	
	# Validate the input pandas file
	assert(os.path.isfile(args.var_file))
	assert(os.path.splitext(args.var_file)[1] == ".pkl")

	return args

def get_paths_dict(args,title_strs,models_dir):
	paths_dict = {}
	for t in title_strs:
		title_str = t
		paths_dict[t] = {}
		current_model_dir = os.path.join(models_dir,title_str)
		
		if args.meta != "":
			current_model_dir = os.path.join(models_dir,
						 args.label,"_%s_%s" % \
						 (t,args.meta))
		if not os.path.isdir(current_model_dir):
			os.makedirs(current_model_dir)
		paths_dict[t]["current_model_dir"] = current_model_dir
		paths_dict[t]["best_model_dir"] = os.path.join(current_model_dir,'model')
		paths_dict[t]["test_predictions_file"] = os.path.join(current_model_dir,
			'test_predictions.json')
		paths_dict[t]["best_model_state"] = os.path.join(current_model_dir,
			'state.json')
		paths_dict[t]["parameters_state"] = os.path.join(current_model_dir,
			'parameters.json')
		paths_dict[t]["np_dir"] = os.path.join(current_model_dir,'npy')
		paths_dict[t]["output_covars_savepath"] = os.path.join(current_model_dir,
						'cache','%s_balanced.pkl'%args.label)
		paths_dict[t]["output_selection_savepath"] = os.path.join(current_model_dir,
						'cache','%s_balanced.npy'%args.label)
		paths_dict[t]["output_test_predictions"] = os.path.join(current_model_dir,
						args.test_predictions_filename)
		paths_dict[t]["output_results"] = os.path.join(current_model_dir,
			'test_results.json')
		paths_dict[t]["output_grad_sample"] = os.path.join(current_model_dir,
			'grad_samples')
		if not os.path.isdir(paths_dict[t]["output_grad_sample"]):
			os.makedirs(paths_dict[t]["output_grad_sample"])
		paths_dict[t]["output_regressor_loss"] = os.path.join(current_model_dir,
			'regressor_loss.png')
		paths_dict[t]["output_encoder_loss"] = os.path.join(current_model_dir,
			'encoder_loss.png')
	return paths_dict


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

def validate_all_vars(all_vars,args):
	for c in args.confounds:
		if c not in all_vars.columns:
			raise Exception("Confound %s not in columns of %s"%(c,args.var_file))
	
	if args.label not in all_vars.columns:
		raise Exception("Label %s not in columns of %s"%(args.label,args.var_file))
	
	for index in all_vars.index:
		if os.path.splitext(index)[1] != ".npy":
			raise Exception(("Indices of %s must all be .npy files: "+\
				"exception at index %s") % (args.var_file,index))


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

