# LMDB file parser

import os,sys
import lmdb
import csv

import numpy as np
import lmdb

from PIL import Image
from scipy.misc import imresize

from eyelibs.lib.keras_lib import lmdb_pb2
from eyelibs.lib.label_embedding.embedding_io import EmbeddingIO

class LMDBWriter():
		
	def __init__(self,file_list,lmdb_dir):
			
		# 1. Path to directory containing data.mdb
		# 2. keep which will take 1000 records from lmdb file into CPU memory

		self.lmdb_dir_path = lmdb_dir
		self.image_file_list = file_list

		self.io = EmbeddingIO(log_dir,'lmdb-writing-{0}'.format(lmdb_dir_path.split('/')[-1]))

		if not os.path.exists(self.lmdb_dir_path):
			self.io.print_error('Could not find LMDB path {0}'.format(self.lmdb_dir_path))
			self.init = False
			return
		#if

		try:
				
			self.lmdb_env = lmdb.open(self.lmdb_dir_path,readonly=True, lock=False)
			self.lmdb_txn = self.lmdb_env.begin()

			self.lmdb_cursor = self.lmdb_txn.cursor()
			self.lmdb_cursor_iterator = self.lmdb_cursor.__iter__()

			self.datum = Datum()
				
		except Exception as e:
				
			self.io.print_error('Error reading LDMB file {0},{1}'.format(self.lmdb_dir_path,e))
			self.init = False
			return
				
		#try
		self.io.print_info('Train LMDB file : {0} '.format(self.lmdb_dir_path))
		self.init = True
		return
			
	#def

	def get_next_slice(self,crop_size=None,mean_value=None,keep=None):
			
		images = []
		labels = []
		
		if keep == None:
			keep = self.keep
		#if
		
		# TODO : HSS - setting fields from YAML
		
		for i in xrange(keep):
			
			# if End of LMDB file start from the begining
			
			if not self.lmdb_cursor.next():
				self.lmdb_cursor.first()
				self.passes += 1
				self.io.print_info('Start of next pass over the training set {0}'.format(self.passes))
			#if
			
			# Read a key-value record from the queue

			key, value = self.lmdb_cursor_iterator.next()

			self.datum.ParseFromString(value)

			arr = np.fromstring(self.datum.data, dtype=np.uint8).reshape(self.datum.channels, self.datum.height, self.datum.width)
			arr = arr.transpose(1,2,0)

			# This casting in required since uint8 and float32 cause an under-flow while mean subtraction below
			arr = np.asarray(arr,dtype=np.float32)
			
			if not crop_size == None:
				arr = np.asarray(imresize(arr,crop_size),dtype=np.float32)
			#if

			if not mean_value == None:
				arr -= mean_value
			#if

			arr = arr.transpose(2,0,1)

			# As theano likes it N_channels x image_H x image_w
			
			lab = self.datum.label
			# Label is a single field but could be an array
			
			images.append(arr)
			labels.append(lab)
				
		#for

		return np.array(images),np.array(labels)

		#def
		
	def check_passes(self):
			
		self.io.print_info('Done {0} passes of lmdb'.format(self.passes))
		
		return self.passes
			
	#def

	def close(self):

		self.lmdb_env.close()

	#def
		
#class