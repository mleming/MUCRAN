#!/usr/bin/python

import os,sys,json,glob
from general_class_balancer import *
import numpy as np
import pandas as pd
from copy import deepcopy as copy
import nibabel as nb
from sklearn.preprocessing import MultiLabelBinarizer
from scipy import ndimage

# Used to get a training set with equal distributions of input covariates
# Can also be used to only have certain ranges of continuous covariates,
# or certain labels of discrete covariates.

working_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

json_label_folder = os.path.join(working_dir,'json','labels')

json_label_filepaths = glob.glob(os.path.join(json_label_folder,'*.json'))

pandas_output = os.path.join(working_dir,'pandas','cache','all_vars.pkl')
json_output   = os.path.join(working_dir,'json','cache','all_vars.json')

image_dim = [96,96,96]

def nifti_to_np(nifti_filepath,output_image_dim):
	nifti_file = nb.load(nifti_filepath)
	nifti_data = nifti_file.get_fdata()
	nifti_data -= nifti_data.min()
	m = nifti_data.max()
	nifti_data = nifti_data / m
	nifti_data = nifti_data.astype(np.float32)
	zp = [output_image_dim[i]/nifti_data.shape[i] for i in range(len(output_image_dim))]
	nifti_data_zoomed = ndimage.zoom(nifti_data,zp)
	return nifti_data_zoomed

def str_to_list(s,nospace=False):
	if s is None or s == "": return []
	if s[0] == "[" and  s[-1] == "]":
		s = s[1:-1]
		s = s.replace("'","").replace("_","").replace("-","")
		if nospace:s=s.replace(" ","")
		s = s.split(",")
		if nospace and "" in s: s.remove("")
		return s
	else:
		return [s]

def iz_nan(k,inc_null_str=False):
	if k is None:
		return True
	if inc_null_str and isinstance(k,str):
		if k.lower() == "null" or k.lower() == "unknown":
			return True
	try:
		if np.isnan(k):
			return True
		else:
			return False
	except:
		if k == np.nan:
			return True
		else:
			return False

def is_list_str(s):
	if iz_nan(s): return False
	return (s[0] == "[" and s[-1] == "]")


def bucketize(arr,n_buckets):
	non_arr_list = []
	max_ = -np.Inf
	min_ = np.Inf
	for i in range(len(arr)):
		if not iz_nan(arr[i]):
			if isinstance(arr[i],str): return arr
			non_arr_list.append(arr[i])
			if arr[i] > max_: max_ = arr[i]
			if arr[i] < min_: min_ = arr[i]
	bucketized_list = np.array(["NaN" for i in range(len(arr))],
			dtype=np.dtype(object))
	non_arr_list = sorted(non_arr_list)
	skips = int(len(non_arr_list)/float(n_buckets)) + 1
	buckets = np.array(non_arr_list[::skips])
	range_dist=((np.arange(n_buckets)/float(n_buckets-1))*(max_-min_))+min_
	buckets = (range_dist + buckets) / 2
	print(buckets)
	#print(max_)
	#print(min_)
	#print(buckets.shape)
	#print(buckets)
	#buckets.append(max_)
	for i in range(len(arr)):
		if not iz_nan(arr[i]):
			for j in range(len(buckets)-1):
				if arr[i] > buckets[j] and \
						arr[i] <= buckets[j+1]:
					bucketized_list[i] = str(j)
					break
	return bucketized_list

# Given the filenames (or, rather, filestubs), returns encoded input and output
# labels, as well as encoded confounds, if specified, as either a set of strings
# or binary arrays
def get_data_from_filenames(filename_list,test_variable=None,confounds=None,
		return_as_strs = False,unique_test_vals = None,all_vars=None,
		return_choice_arr = False,dict_obj=None,return_as_dict=False,
		append_title=False):
	uniques = None
	if dict_obj is not None:
		if "uniques" in dict_obj:
			uniques = dict_obj["uniques"]
	if all_vars is None and test_variable is not None:
		all_vars = pd.read_pickle(pandas_output)
	if append_title:
		X_filenames_list = [os.path.join(working_dir,
	       "%s_resized_%d.npy"%(_[1:],image_dim[0])) for _ in filename_list]
	else:
		X_filenames_list = filename_list
	selection = np.array([os.path.isfile(_) for _ in X_filenames_list],
			dtype=bool)
	if confounds is not None:
		confound_strs = [[None for _ in confounds] \
					for __ in filename_list]
	Y_strs = [None for _ in filename_list]
	X = np.zeros((len(filename_list),
		image_dim[0],image_dim[1],image_dim[2]))
	for i in range(len(filename_list)):
		if selection[i] == 0: continue
		f = X_filenames_list[i]
		f_key = filename_list[i]
		assert(os.path.isfile(f))
		try:
			X_single = np.load(f)
		except:
			selection[i] = 0
			continue
		X[i,:,:,:] = X_single
		if test_variable is not None:
			Y_strs[i] = str_to_list(all_vars.loc[f_key,test_variable],
				nospace=True)
			if confounds is not None:
				for j in range(len(confounds)):
					confound_strs[i][j] = all_vars.loc[f_key,
						confounds[j]]
	filename_list = list(np.array(filename_list)[selection])
	X_filenames_list = list(np.array(X_filenames_list)[selection])
	X = X[selection,:,:,:]
	if test_variable is None:
		return X
	Y_strs = list(np.array(Y_strs)[selection])
	if return_as_strs:
		if confounds is not None:
			return X_filenames_list,Y_strs,confound_strs
		else:
			return X_filenames_list,Y_strs
	
	mlb = MultiLabelBinarizer()
	if unique_test_vals is not None:
		mlb.fit([unique_test_vals])
	else:
		Y_strs_all = []
		for s in all_vars.loc[:,test_variable]:
			if not iz_nan(s):
				Y_strs_all.append(str_to_list(s,nospace=True))
		mlb.fit(Y_strs_all)
	Y = mlb.transform(Y_strs)
	if confounds is not None:
		n_buckets = 3
		if uniques is None or np.any([c not in uniques for c in confounds]):
			uniques = {}
			for c in confounds:
				uniques[c] = {}
				lis = list(all_vars.loc[:,c])
				if np.any([isinstance(_,str) for _ in lis]):
					uniques[c]["discrete"] = True
					u = set()
					for l in lis:
						if not iz_nan(l):
							u.add(l)
					u = sorted(list(u))
					uniques[c]["unique"] = u
					n_buckets = max(n_buckets,len(u))
				else:
					uniques[c]["discrete"] = False
					max_ = -np.inf
					min_ = np.inf
					nonnan_list = []
					for l in lis:
						if not iz_nan(l):
							max_ = max(max_,l)
							min_ = min(min_,l)
							nonnan_list.append(l)
					uniques[c]["max"] = max_
					uniques[c]["min"] = min_
					uniques[c]["nonnan_list"] = sorted(nonnan_list)
			for c in confounds:
				if not uniques[c]["discrete"]:
					n_buckets_cont = min(n_buckets,10)
					skips = int(len(uniques[c]["nonnan_list"])/\
						float(n_buckets_cont)) + 1
					uniques[c]["nonnan_list"] = \
						uniques[c]["nonnan_list"][::skips]
					# Get mean between density and range dists
					if True:
						max_ = uniques[c]["max"]
						min_ = uniques[c]["min"]
						rd = np.arange(n_buckets_cont)
						rd = rd / float(n_buckets_cont-1)
						rd = list((rd * (max_ - min_)) + min_)
						uniques[c]["nonnan_list"] = \
							[(rd[i] + \
							uniques[c]["nonnan_list"][i])/2 \
							for i in range(n_buckets_cont)]
						uniques[c]["nonnan_list"][-1] = max_
						uniques[c]["nonnan_list"][0] = min_
					assert(len(uniques[c]["nonnan_list"]) == \
						n_buckets_cont)
		confound_encode = np.zeros((len(filename_list),len(confounds),
				n_buckets + 1))
		if return_choice_arr:
			choice_arr = np.zeros((1,len(confounds),n_buckets + 1))
			for i in range(choice_arr.shape[1]):
				choice_arr[:,i,-1] = 1
				c = confounds[i]
				if uniques[c]["discrete"]:
					c_uniques = uniques[c]["unique"]
					for j in range(len(c_uniques)):
							choice_arr[:,i,j] = 1
				else:
					choice_arr[:,i,:] = 1
		for j in range(len(confounds)):
			c = confounds[j]
			if uniques[c]["discrete"]:
				
				c_uniques = uniques[c]["unique"]
				for i in range(len(filename_list)):
					if iz_nan(confound_strs[i][j]):
						confound_encode[i,j,-1] = 1
					else:
						confound_encode[i,j,
							c_uniques.index(\
							confound_strs[i][j])]=1
			else:
				max_ = uniques[c]["max"]
				min_ = uniques[c]["min"]
				for i in range(len(filename_list)):
					if iz_nan(confound_strs[i][j]):
						confound_encode[i,j,-1] = 1
					else:
						unnl = uniques[c]["nonnan_list"]
						for kk in range(len(unnl)-1):
							if unnl[kk] <= confound_strs[i][j] and \
								unnl[kk+1] >= confound_strs[i][j]:
								confound_encode[i,j,kk]=1
								break
		try:
			assert(np.all(np.sum(confound_encode,axis=2) == 1))
		except:
			print(np.sum(confound_encode,axis=2))
			print("Assertion failed")
			print(confound_encode)
			exit()
		if return_as_dict:
			obj = {}
			obj["X"] = X
			obj["Y"] = Y
			obj["confound_encode"] = confound_encode
			if "choice_arr" in locals():
				obj["choice_arr"] = choice_arr
			obj["classes"] = list(mlb.classes_)
			obj["uniques"] = uniques
			return obj
		elif return_choice_arr:
			return X,Y,confound_encode,choice_arr,list(mlb.classes_)
		else:
			return X,Y,confound_encode
	else:
		return X,Y

def get_table_data(pandas_output,json_output):
	if os.path.isfile(pandas_output) and os.path.isfile(json_output):
		covars = json.load(open(json_output,'r'))
		covars_df = pd.read_pickle(pandas_output)
	else:
		columns = [os.path.splitext(os.path.basename(filepath))[0] \
			for filepath in json_label_filepaths]
		covars = {}
		
		emptydict = {}
		for c in columns:
			emptydict[c] = None
		
		for filepath in json_label_filepaths:
			json_dict = json.load(open(filepath,'r'))
			key = os.path.splitext(os.path.basename(filepath))[0]
			if json_dict["discrete"]:
				c = 0
				ils = False
				for value in json_dict[key]: 
					if not iz_nan(value):
						ils = is_list_str(value)
						break
				if ils and False:
					for value in json_dict[key]:
						values = str_to_list(value,
							nospace=True)
						for dataset_name in \
							 json_dict[key][value]:
							if dataset_name not in \
								covars:
								covars[dataset_name] = copy(emptydict)
							for v in values:
								sname = "%s_%s" % (key,value)
								if sname not in emptydict:
									emptydict[sname] = 0
									for dn in covars:
										covars[dn][sname] = 0
								covars[dataset_name][sname] = 1
				for value in json_dict[key]:
					for dataset_name in json_dict[key][value]:
						if dataset_name not in covars:
							covars[dataset_name] = copy(emptydict)
						covars[dataset_name][key] = value
						c += 1
			else:
				c = 0
				for v in json_dict[key]:
					value,dataset_name = v
					if dataset_name not in covars:
						covars[dataset_name] = copy(emptydict)
					covars[dataset_name][key] = value
					c += 1
		covars_df = pd.DataFrame.from_dict(covars,orient='index',columns = columns)
		covars_df.to_pickle(pandas_output)
		str_to_list_stor = {}
		for index in covars_df.index:
			for icd in ["ICD","ICD_partial"]:
				s = covars_df.loc[index,icd]
				if s not in str_to_list_stor:
					str_to_list_stor[s] = \
						str_to_list(s,nospace=True)
				for value in str_to_list_stor[s]:
					sname = "%s_%s" % (icd,value)
					if sname not in covars_df.columns:
						covars_df[sname] = "0"
					covars_df.loc[index,sname] = "1"
		covars_df.to_pickle(pandas_output)
		json.dump(covars,open(json_output,'w'),indent=4)
	return covars_df,covars


from copy import deepcopy as copy

def recompute_selection_ratios(selection_ratios,selection_limits,N):
	new_selection_ratios = copy(selection_ratios)
	assert(np.any(np.isinf(selection_limits)))
	variable = [True for i in range(len(selection_ratios))]

	for i in range(len(selection_ratios)):
		if selection_ratios[i] * N > selection_limits[i]:
			new_selection_ratios[i] = selection_limits[i] / N
			variable[i] = False
		else:
			new_selection_ratios[i] = selection_ratios[i]
	vsum = 0.0
	nvsum = 0.0
	for i in range(len(selection_ratios)):
		if variable[i]: vsum += new_selection_ratios[i]
		else: nvsum += new_selection_ratios[i]
	assert(nvsum < 1)
	for i in range(len(selection_ratios)):
		if variable[i]:
			new_selection_ratios[i] = \
				(new_selection_ratios[i] / vsum) * (1 - nvsum)
	return new_selection_ratios

def get_balanced_filename_list(test_variable,confounds_array,
		selection_ratios = [0.66,0.16,0.16],
		selection_limits = [np.Inf,np.Inf,np.Inf],
		pandas_output = pandas_output,json_output = json_output,
		value_ranges = [],output_covars_savepath = None,
		output_selection_savepath = None,test_value_ranges=None,
		get_all_test_set=False,total_size_limit=None,
		verbose=False,non_confound_value_ranges = {},all_vars = None,
		n_buckets = 10):
	if len(value_ranges) == 0:
		value_ranges = [None for _ in confounds_array]
	assert(len(value_ranges) == len(confounds_array))
	
	if all_vars is None:
		covars_df,covars = get_table_data(pandas_output,json_output)
	else:
		covars_df = all_vars
	if verbose: print("len(covars): %d" % len(covars_df))
	value_selection = np.ones((len(covars_df),),dtype=bool)
	for ncv in non_confound_value_ranges:
		assert(ncv not in confounds_array)
		confounds_array.append(ncv)
		value_ranges.append(non_confound_value_ranges[ncv])
	confounds_array.append(test_variable)
	value_ranges.append(test_value_ranges)
	if verbose: print("confounds_array: %s" % str(confounds_array))
	if verbose: print("value_ranges: %s" % str(value_ranges))
	for i in range(len(confounds_array)):
		temp_value_selection = np.zeros((len(covars_df),),dtype=bool)
		c = covars_df[confounds_array[i]]
		value_range = value_ranges[i]
		if value_range is None:
			continue
		if isinstance(value_range,tuple):
			for j in range(len(c)):
				if c[j] is None:
					continue
				if c[j] >= value_range[0] and\
						 c[j] <= value_range[1]:
					temp_value_selection[j] = True
		elif callable(value_range):
			for j in range(len(c)):
				if c[j] is None:
					continue
				if value_range(c[j]):
					temp_value_selection[j] = True
		else:
			for j in range(len(c)):
				if c[j] is None:
					continue
				if c[j] in value_range:
					temp_value_selection[j] = True	
		value_selection = np.logical_and(value_selection,
					temp_value_selection)
	del confounds_array[-1]
	del value_ranges[-1]
	for ncv in non_confound_value_ranges:
		del confounds_array[-1]
		del value_ranges[-1]
	if verbose:
		print("value_selection.sum(): %s"%str(value_selection.sum()))
	if verbose:
		print("value_selection.shape: %s"%str(value_selection.shape))
	covars_df = covars_df[value_selection]
	covars_df = covars_df.sample(frac=1)
	test_vars = covars_df[test_variable].to_numpy(dtype=np.dtype(object))
	# If it's a string array, it just returns strings
	test_vars = bucketize(test_vars,n_buckets)
	ccc = {}
	#for t in test_vars:
	#	if t not in ccc: ccc[t] = 0
	#	ccc[t] += 1
	#for t in ccc: print("%s: %d" % (t,ccc[t]))
	#assert(np.all([isinstance(i,str) for i in test_vars]))
	if output_selection_savepath is not None and \
			os.path.isfile(output_selection_savepath):
		selection = np.load(output_selection_savepath)
	else:
		
		if len(confounds_array) == 0:
			if verbose: print(test_value_ranges)
			#blanks = np.array(["" for _ in range(len(test_vars))])
			#blanks = np.reshape(blanks,(1,blanks.shape[0]))
			selection = class_balance(test_vars,[],
				unique_classes=test_value_ranges,plim=0.1)
			#selection = np.ones(test_vars.shape)
		else:
			selection = class_balance(test_vars,
				covars_df[confounds_array].to_numpy(\
					dtype=np.dtype(object)).T,
				unique_classes=test_value_ranges,plim=0.1)
		if output_covars_savepath is not None:
			if not os.path.isdir(\
				os.path.dirname(output_covars_savepath)):
				os.makedirs(os.path.dirname(\
					output_covars_savepath))
			covars_df[selection].to_pickle(output_covars_savepath)
		selection_ratios = recompute_selection_ratios(selection_ratios,
			selection_limits,np.sum(selection))
		if total_size_limit is not None:
			select_sum = selection.sum()
			rr = list(range(len(selection)))
			for i in rr:
				if select_sum <= total_size_limit:
					break
				if selection[i]:
					selection[i] = 0
					select_sum -= 1
		selection = separate_set(selection,selection_ratios,
			covars_df["PatientID"].to_numpy(dtype=\
			np.dtype(object)).T)
		if output_selection_savepath is not None:
			np.save(output_selection_savepath,selection)
	all_files = (covars_df.index.values)
	if get_all_test_set:
		selection[selection == 0] = 2
	X_files = [all_files[selection == i] \
			for i in range(1,len(selection_ratios) + 1)]
	Y_files = [test_vars[selection == i] \
			for i in range(1,len(selection_ratios) + 1)]
	if verbose: print(np.sum([len(x) for x in X_files]))
	for i in range(len(X_files)):
		rr = list(range(len(X_files[i])))
		random.shuffle(rr)
		X_files[i] = X_files[i][rr]
		Y_files[i] = Y_files[i][rr]
	return X_files,Y_files

def get_balanced_mri_data(test_variable = "Test_Control",
	confounds_array = ["Ages","SexDSC","BodyPartExamined"],
	selection_ratios = [0.66,0.16,0.16], pandas_output = pandas_output,
	json_output = json_output,image_dim = (96,96,96),value_ranges = []):
	y_list,X_filebase_list = get_balanced_filename_list(test_variable,
						confounds_array,
						selection_ratios,
						pandas_output = pandas_output,
						json_output = json_output,
						value_ranges=value_ranges) 
	X_filenames_list = [[ os.path.join(working_dir,
		"%_resized_%d.npy" % (f,image_dim[0])) for f in x] \
		for x in X_filenames_list]
	X = [np.zeros(tuple([len(x)] + list(image_dim))) \
		for x in X_filenames_list]
	for i in range(len(X)):
		for j in range(X[i].shape[0]):
			X[i][j,:,:,:]=nb.load(X_filenames_list[i][j]).get_data()
	return X,y_list

#confounds_array = ["Ages","SexDSC","BodyPartExamined"]
#confounds_array = ["SexDSC","BodyPartExamined","DeviceSerialNumber","Ages"]
#confounds_array = ["SexDSC","BodyPartExamined"]

if __name__ == '__main__':
	[X_files],_ = get_balanced_filename_list("Ages",
		confounds_array = [],
		selection_ratios = [1],pandas_output = pandas_output,
		json_output = json_output,n_buckets=3)
	#print(len(X_files[0]))
	#X,Y,C = get_data_from_filenames(X_files[0:500],"SexDSC",confounds=["SexDSC","SequenceVariant","Ages","RepetitionTime","EchoTime"])


