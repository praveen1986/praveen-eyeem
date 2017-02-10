# Hashnet on AlexNet definition in keras

import sys,os
import yaml
import json
import numpy as np

from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten, Merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.embeddings import Embedding
from keras.optimizers import SGD

from rnd_libs.lib.label_embedding.embedding_io import EmbeddingIO
from rnd_libs.lib.keras.loss_functions import ObjectiveFunctions

# AlexNet for Hashing with batch normalization in Keras
# input image is 224x224

class AlexHash():

	def __init__(self):

		self.config_file = None
		self.io = EmbeddingIO(None)
		self.init = False
		self.cfgs = None

	#def

	def configure(self,config_file):

		self.config_file = config_file
		self.init = False

		if not os.path.exists(self.config_file):
			self.io.print_error('Could not find config file for VGG-19 {0}'.format(self.config_file))
			self.init = False
			return
		#if

		pfile = open(self.config_file,'r')
		self.cfgs = yaml.load(pfile)
		pfile.close()

		self.init = True

	#def

	def define(self):

		#
		# TODO : get the filter size, number of filters, reception field size from yaml file
		#

		try:

			self.model = Graph()

			self.model.add_input(name='img', input_shape=(self.cfgs['n_channels'], self.cfgs['image_height'], self.cfgs['image_width']))

			# C1
			self.model.add_node(Convolution2D(96, 11, 11,subsample=(4,4)),name='conv1',input='img')
			self.model.add_node(Activation(self.cfgs['conv_non_linearity']),name='relu1',input='conv1')
			#self.model.add_node(LRN2D(alpha=0.0001, k=1, beta=0.75, n=5),name='norm1', input='relu1')
			self.model.add_node(MaxPooling2D(pool_size=(3, 3),stride=(2,2)),name='pool1',input='norm1')
			# C1

			# C2
			self.model.add_node(ZeroPadding2D((2,2)),name='zpad5',input='pool1')
			self.model.add_node(Convolution2D(256, 5, 5),name='conv2',input='zpad5')
			self.model.add_node(Activation(self.cfgs['conv_non_linearity']),name='relu2',input='conv2')
			#self.model.add_node(LRN2D(alpha=0.0001, k=1, beta=0.75, n=5),name='norm2', input='relu2')
			self.model.add_node(MaxPooling2D(pool_size=(3, 3),stride=(2,2)),name='pool2',input='norm2')
			# C2

			# C3
			self.model.add_node(ZeroPadding2D((1,1)),name='zpad9',input='pool2')
			self.model.add_node(Convolution2D(384, 3, 3),name='conv3',input='zpad9')
			self.model.add_node(Activation(self.cfgs['conv_non_linearity']),name='relu3',input='conv3')
			# C3

			# C4
			self.model.add_node(ZeroPadding2D((1,1)),name='zpad11',input='relu3')
			self.model.add_node(Convolution2D(384, 3, 3),name='conv4',input='zpad11')
			self.model.add_node(Activation(self.cfgs['conv_non_linearity']),name='relu4',input='conv4')
			# C4

			# C5
			self.model.add_node(ZeroPadding2D((1,1)),name='zpad13',input='relu4')
			self.model.add_node(Convolution2D(256, 3, 3),name='conv5',input='zpad13')
			self.model.add_node(Activation(self.cfgs['conv_non_linearity']),name='relu5',input='conv5')
			self.model.add_node(MaxPooling2D(pool_size=(3, 3),stride=(2,2)),name='pool5',input='relu5')
			# C5

			# Flatten - flat16 since its the 16th layer errr.
			self.model.add_node(Flatten(),name='flat16',input='pool5')
			# Flatten

			# FC6
			self.model.add_node(Dense(4096,init=self.cfgs['fc_init']),name='fc6',input='flat16')
			self.model.add_node(Activation(self.cfgs['fc_non_linearity']),name='relu6',input='fc6')
			self.model.add_node(Dropout(self.cfgs['drop_out']),name='drop6',input='relu6')
			# FC6

			# FC7
			self.model.add_node(Dense(4096,init=self.cfgs['fc_init']),name='fc7',input='drop6')
			self.model.add_node(Activation(self.cfgs['fc_non_linearity']),name='relu7',input='fc7')
			self.model.add_node(Dropout(self.cfgs['drop_out']),name='drop7',input='relu7')
			# FC7

			# Contact fc6, fc7
			self.model.add_node(Activation('linear'),name='concat',merge_mode='concat',inputs=['drop6','drop7'])
			#

			# Hash Layer
			self.model.add_node(Dense(self.cfgs['n_bits']),name='hash',input='concat')
			# Hash layer

			# Last Aktivation
			self.model.add_node(Activation(self.cfgs['activation']),name='bits',input='hash')
			# Totally last Aktivation

			self.model.add_output(name='output',input='bits')

			if not self.cfgs['model_weights_file'] == 'None':
				self.io.print_info('Porting weights from {0}'.format(self.cfgs['model_weights_file']))
				self.init_from_this()
			#if

		except Exception as e:

			self.io.print_error('Error configuring the model, {0}'.format(e))
			self.init = False
			return

		#try

		self.init = True

	#def

	def init_from_this(self):

		weights_file = self.cfgs['model_weights_file']

		if weights_file.endswith('.npz'):
			self.load_weights(weights_file)
			self.io.print_info('Weights initalized from {0}'.format(weights_file))
		#if

		if weights_file.endswith('.caffemodel'):
			self.io.print_warning('Loading from caffe model not implemented starting with random weights')
		#if

	#def

	def load_weights(self, filepath):

		'''
			This method does not make use of Graph.set_weights()
			for backwards compatibility. Has been modified to fit a graphical model
		'''

		# Loads weights from npz file

		import numpy as np

		pfile = open(filepath,'r')

		p = np.load(pfile)
		params = p['params'].item()
		pfile.close()

		for key, node in self.model.nodes.items():

			if key in self.cfgs['ignore_nodes']:
				self.io.print_info('Skipping weights transfer from {0},{1}'.format(key,node.get_config()['name']))
			#if

			if key in params.keys():
				w = params[key]
				node.set_weights(w)
				self.io.print_info('Transferring weights from {0},{1}'.format(key,node.get_config()['name']))
			#if

		#for

	#def

	def compile(self,compile_cfgs):

		try:
			sgd = SGD(lr=compile_cfgs['lr'], decay=compile_cfgs['decay'], momentum=compile_cfgs['momentum'], nesterov=True)
			self.model.compile(loss=loss_functions_dict[compile_cfgs['loss']], optimizer=sgd)
		except Exception as e:
			self.io.print_error('Error configuring the model, {0}'.format(e))
			self.init = False
			return
		#try

		self.init = True

	#def

#class
