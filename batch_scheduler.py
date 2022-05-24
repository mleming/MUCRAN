import os
import torchvision.transforms as transforms
from base_dataset import BaseDataset
from pdb import set_trace as st
import random
import numpy as np
import torch
import pandas as pd
import torch.multiprocessing
from utils import *
torch.multiprocessing.set_sharing_strategy('file_system')

class BatchScheduler(BaseDataset):
	def __init__(self,args,all_vars,val_ranges,path_func = None):
		self.args = args
		self.batch_size = self.args.batch_size
		self.all_vars = all_vars
		self.file_list_dict = {}
		self.val_ranges = val_ranges
		self.label = self.args.label
		self.confounds = self.args.confounds
		self.path_func = path_func
		if self.path_func is None:
			self.path_func = lambda k: k
		ind = self.path_func(self.all_vars.index[0])
		self.in_dim = np.load(ind).shape
		X,Y,C,choice_arr,output_label_and_confound = get_data_from_filenames(\
				[ind],args.label,confounds=self.confounds,
				return_as_strs = False,unique_test_vals = None,
				all_vars=all_vars,return_choice_arr=True,return_as_dict=False,
				dict_obj=None)
		YC,YC_dud = YC_conv(Y,C,self.args.y_weight)
		self.out_dim = YC.shape
	def __getitem__(self, index):
		if index > len(self): raise StopIteration
		if self.label not in self.file_list_dict or len(self.file_list_dict[self.label]) == 0:
			[X_files],_ = get_balanced_filename_list(self.label,
				self.args.match_confounds,
				selection_ratios=[1], total_size_limit=self.args.total_load,
				non_confound_value_ranges = self.val_ranges,verbose=False,
				all_vars=self.all_vars)
			random.shuffle(X_files)
			self.file_list_dict[self.label] = X_files
		
		match_conf = self.confounds[index % len(self.confounds)]
		if match_conf not in self.file_list_dict or len(self.file_list_dict[match_conf]) == 0:
			[X_files],_ = get_balanced_filename_list(match_conf,
				self.args.match_confounds,
				selection_ratios=[1], total_size_limit=self.args.total_load,
				non_confound_value_ranges = self.val_ranges,verbose=False,
				all_vars=self.all_vars)
			if len(X_files) < self.batch_size:
				[_X_files],_ = get_balanced_filename_list(self.label,
					self.args.match_confounds,
					selection_ratios=[1], total_size_limit=self.args.total_load,
					non_confound_value_ranges = self.val_ranges,verbose=False,
					all_vars=self.all_vars)
				X_files = X_files + _X_files
			random.shuffle(X_files)
			self.file_list_dict[match_conf] = X_files

		rlist = np.concatenate((self.file_list_dict[self.label][:int(self.batch_size / 2)],
				self.file_list_dict[match_conf][:int(self.batch_size / 2)]),axis=0)	
		self.file_list_dict[self.label] = self.file_list_dict[self.label][int(self.batch_size / 2):]	
		self.file_list_dict[match_conf] = self.file_list_dict[match_conf][int(self.batch_size / 2):]
	
		X,Y,C,choice_arr,output_label_and_confound = get_data_from_filenames(\
				rlist,self.args.label,confounds=self.confounds,
				return_as_strs = False,unique_test_vals = None,
				all_vars=self.all_vars,return_choice_arr=True,return_as_dict=False,
				dict_obj=None)
		YC,YC_dud = YC_conv(Y,C,self.args.y_weight)
		return X,Y,YC,YC_dud

	def __len__(self):
		return self.args.total_load // self.batch_size

	def name(self):
		return 'BatchScheduler'
