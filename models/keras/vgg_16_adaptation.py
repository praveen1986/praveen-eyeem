# VGG definition in keras

import sys,os
import yaml
import json

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD

from rnd_libs.lib.label_embedding.embedding_io import EmbeddingIO

#AlexNet with batch normalization in Keras 
#input image is 224x224

class VGG16Adap():

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
			self.model = Sequential()

			# C1

			self.model.add(ZeroPadding2D((1,1),input_shape=(self.cfgs['n_channels'], self.cfgs['image_height'], self.cfgs['image_width'])))
			self.model.add(Convolution2D(64, 3, 3,activation = self.cfgs['conv_non_linearity']))

			self.model.add(ZeroPadding2D((1,1)))
			self.model.add(Convolution2D(64, 3, 3,init=self.cfgs['conv_init'],activation = self.cfgs['conv_non_linearity']))
			self.model.add(MaxPooling2D(pool_size=(2, 2),stride=(2,2)))
			# C1

			# C2
			self.model.add(ZeroPadding2D((1,1)))
			self.model.add(Convolution2D(128, 3, 3,init=self.cfgs['conv_init'],activation = self.cfgs['conv_non_linearity']))

			self.model.add(ZeroPadding2D((1,1)))
			self.model.add(Convolution2D(128, 3, 3,init=self.cfgs['conv_init'],activation = self.cfgs['conv_non_linearity']))

			self.model.add(MaxPooling2D(pool_size=(2, 2),stride=(2,2)))
			# C2

			# C3
			self.model.add(ZeroPadding2D((1,1)))
			self.model.add(Convolution2D(256, 3, 3,init=self.cfgs['conv_init'],activation = self.cfgs['conv_non_linearity']))

			self.model.add(ZeroPadding2D((1,1)))
			self.model.add(Convolution2D(256, 3, 3,init=self.cfgs['conv_init'],activation = self.cfgs['conv_non_linearity']))

			self.model.add(ZeroPadding2D((1,1)))
			self.model.add(Convolution2D(256, 3, 3,init=self.cfgs['conv_init'],activation = self.cfgs['conv_non_linearity']))

			self.model.add(MaxPooling2D(pool_size=(2, 2),stride=(2,2)))
			# C3

			# C4
			self.model.add(ZeroPadding2D((1,1)))
			self.model.add(Convolution2D(512, 3, 3,init=self.cfgs['conv_init'],activation = self.cfgs['conv_non_linearity']))

			self.model.add(ZeroPadding2D((1,1)))
			self.model.add(Convolution2D(512, 3, 3,init=self.cfgs['conv_init'],activation = self.cfgs['conv_non_linearity']))

			self.model.add(ZeroPadding2D((1,1)))
			self.model.add(Convolution2D(512, 3, 3,init=self.cfgs['conv_init'],activation = self.cfgs['conv_non_linearity']))

			self.model.add(MaxPooling2D(pool_size=(2, 2),stride=(2,2)))
			# C4

			# C5
			self.model.add(ZeroPadding2D((1,1)))
			self.model.add(Convolution2D(512, 3, 3,init=self.cfgs['conv_init'],activation = self.cfgs['conv_non_linearity']))

			self.model.add(ZeroPadding2D((1,1)))
			self.model.add(Convolution2D(512, 3, 3,init=self.cfgs['conv_init'],activation = self.cfgs['conv_non_linearity']))

			self.model.add(ZeroPadding2D((1,1)))
			self.model.add(Convolution2D(512, 3, 3,init=self.cfgs['conv_init'],activation = self.cfgs['conv_non_linearity']))

			self.model.add(MaxPooling2D(pool_size=(2, 2),stride=(2,2)))
			# C5

			self.model.add(Flatten())

			# FC6
			self.model.add(Dense(4096,init=self.cfgs['fc_init'],activation=self.cfgs['fc_non_linearity']))
			# FC6

			# FC7
			self.model.add(Dropout(self.cfgs['drop_out']))
			self.model.add(Dense(4096,init=self.cfgs['fc_init'],activation=self.cfgs['fc_non_linearity']))
			# Fc7

			# FC8
			self.model.add(Dropout(self.cfgs['drop_out']))
			self.model.add(Dense(self.cfgs['nb_classes'],init=self.cfgs['fc_init'],activation=self.cfgs['activation']))
			# FC8

			if not self.cfgs['model_weights_file'] == None:
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

		if weights_file.endswith('.hdf5'):
			self.load_weights(weights_file)
			self.io.print_info('Weights initalized from {0}'.format(weights_file))
		#if

		if weights_file.endswith('.caffemodel'):
			self.io.print_warning('Loading from caffe model not implemented starting with random weights')
		#if

	#def

	def load_weights(self, filepath):

		'''
			This method does not make use of Sequential.set_weights()
			for backwards compatibility.
		'''

		# Loads weights from HDF5 file
		
		import h5py

		f = h5py.File(filepath)

		for k in range(f.attrs['nb_layers']):

			layer_type = self.model.layers[k].get_config()['name']
			layer_name_string = '{0}_layer_{1}'.format(layer_type,k)

			if layer_name_string in self.cfgs['ignore_while_loading']:
				self.io.print_warning('Ignoring weights from {0}'.format(layer_name_string))
				continue
			#if

			layer_name = 'layer_{}'.format(k)
			g = f[layer_name]
			self.io.print_info('Transfering weights from {0}'.format(layer_name_string))

			weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
			self.model.layers[k].set_weights(weights)

		#for

		f.close()

		self.io.print_info(self.model.layers[-1].get_config().values())

	#def

	def compile(self,compile_cfgs):

		try:
			sgd = SGD(lr=compile_cfgs['lr'], decay=compile_cfgs['decay'], momentum=compile_cfgs['momentum'], nesterov=True)
			self.model.compile(loss=compile_cfgs['loss'], optimizer=sgd)
		except Exception as e:
			self.io.print_error('Error configuring the model, {0}'.format(e))
			self.init = False
			return
		#try

		self.init = True

	#def

#class