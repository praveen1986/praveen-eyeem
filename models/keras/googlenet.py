# GoogLeNet definition in python

import numpy as np

import sys,os
import yaml
import json
import h5py

from keras.optimizers import SGD, RMSprop, Adagrad, Adam
from keras.layers.core import Merge
from keras.models import Sequential, Graph
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation, Dense, Reshape, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D

from rnd_libs.lib.label_embedding.embedding_io import EmbeddingIO
from rnd_libs.lib.keras.loss_functions import ObjectiveFunctions

layer_dict = {'Convolution2D': Convolution2D,
							'ZeroPadding2D': ZeroPadding2D,
							'MaxPooling2D':MaxPooling2D,
							'BatchNormalization':BatchNormalization,
							'Activation': Activation}

pooling_dict = {'max':MaxPooling2D,'average':AveragePooling2D}
optimizer_dict = {'sgd':SGD,'rmsprop':RMSprop, 'adam':Adam}

class GoogleNet():

	def __init__(self):

		self.layer_list = []
		self.config_file = None
		self.io = EmbeddingIO(None)
		self.init = False
		self.cfgs = None
		self.loss_functions = ObjectiveFunctions()

	#def

	def configure(self,config_file):

		self.config_file = config_file
		self.init = False

		if not os.path.exists(self.config_file):
			self.io.print_error('Could not find config file for GoogLeNet {0}'.format(self.config_file))
			self.init = False
			return
		#if

		pfile = open(self.config_file,'r')
		self.cfgs = yaml.load(pfile)
		pfile.close()

		self.init = True

	#def

	def add_to_graph(self, *args, **kwargs):

		self.model.add_node(*args, **kwargs)
		self.last_added_node = kwargs['name']

		self.layer_list.append(kwargs['name'])

		return kwargs['name']

	#def

	def add_bn_conv_layer(self, *args, **kwargs):

		layer_name = kwargs['name']
		input_layer = kwargs['input']

		del kwargs['name']
		del kwargs['input']

		kwargs['border_mode'] = 'same'

		if 'padding' in kwargs:
			layer_name = layer_name + '_pad'
			self.add_to_graph(ZeroPadding2D(padding=kwargs['padding']), name=layer_name, input=input_layer)
			input_layer = layer_name
			del kwargs['padding']
		#if

		# CONV with linear activation by default
		layer_name = layer_name + '_conv'
		self.add_to_graph(Convolution2D(*args, **kwargs), name=layer_name, input=input_layer)

		# Batch normalization added directly on output of a linear layer
		input_layer = layer_name
		layer_name = layer_name + '_bn'
		_ = self.add_to_graph(BatchNormalization(mode=0, epsilon=0.0001, axis=1), name=layer_name, input=input_layer)

		# Standard normalization
		input_layer = layer_name
		layer_name = layer_name + '_nonlin'
		_ = self.add_to_graph(Activation('relu'), name=layer_name, input=input_layer)

		return layer_name

	#def

	def add_inception(self, input_layer, list_nb_filter, base_name):

		tower_1_1 = self.add_bn_conv_layer(name=base_name+'tower_1_1', input=input_layer, nb_filter=list_nb_filter[0], nb_row=1, nb_col=1)

		tower_2_1 = self.add_bn_conv_layer(name=base_name+'tower_2_1', input=input_layer, nb_filter=list_nb_filter[1], nb_row=1, nb_col=1)
		tower_2_2 = self.add_bn_conv_layer(name=base_name+'tower_2_2', input=tower_2_1, nb_filter=list_nb_filter[2], nb_row=3, nb_col=3)

		tower_3_1 = self.add_bn_conv_layer(name=base_name+'tower_3_1', input=input_layer, nb_filter=list_nb_filter[3], nb_row=1, nb_col=1)
		tower_3_2 = self.add_bn_conv_layer(name=base_name+'tower_3_2', input=tower_3_1, nb_filter=list_nb_filter[4], nb_row=5, nb_col=5)

		tower_4_1 = self.add_to_graph(MaxPooling2D((3, 3), strides=(1, 1), border_mode='same'), name=base_name+'tower_4_1', input=input_layer)
		tower_4_2 = self.add_bn_conv_layer(name=base_name+'tower_4_2', input=tower_4_1, nb_filter=list_nb_filter[5], nb_row=1, nb_col=1)

		self.add_to_graph(Activation("linear"), name=base_name, inputs=[tower_1_1,tower_2_2,tower_3_2,tower_4_2], merge_mode="concat", concat_axis=1)

		self.io.print_info('Added Inception {0}'.format(base_name))

	def define(self):

		try:

			self.model = Graph()
			self.model.add_input(name='input', input_shape=(self.cfgs['n_channels'], self.cfgs['image_height'], self.cfgs['image_width']))

			pad_1 = self.add_to_graph(ZeroPadding2D((3,3)), name='pad_1',input='input')
			conv_1 = self.add_bn_conv_layer(name='conv_1', input=self.last_added_node, nb_filter=64, nb_row=7, nb_col=7,subsample=(2,2))

			pad_2 = self.add_to_graph(ZeroPadding2D((1,1)), name='pad_2',input=self.last_added_node)
			max_1 = self.add_to_graph(MaxPooling2D(pool_size=(3,3), strides=(2,2)), name='max_1', input=self.last_added_node)

			pad_3 = self.add_to_graph(ZeroPadding2D((1,1)), name='pad_3', input=self.last_added_node)
			conv_2 = self.add_bn_conv_layer( name='conv_2', input=self.last_added_node, nb_filter=192, nb_row=3, nb_col=3 )

			pad_4 = self.add_to_graph(ZeroPadding2D((1,1)), name='pad_4', input=self.last_added_node)
			max_2 = self.add_to_graph(MaxPooling2D(pool_size=(3,3), strides=(2,2)), name='max_2', input=self.last_added_node)

			#Inception_layer 1
			nb_kernel_inception_1=[64,96,128,16,32,32]
			self.add_inception(self.last_added_node,nb_kernel_inception_1,'inception_1')

			#Inception_layer 2
			nb_kernel_inception_2=[128,128,192,32,96,64]
			self.add_inception(self.last_added_node,nb_kernel_inception_2,'inception_2')

			pad_5 = self.add_to_graph(ZeroPadding2D((1,1)), name='pad_5', input=self.last_added_node)
			max_3 = self.add_to_graph(MaxPooling2D(pool_size=(3,3), strides=(2,2)), name='max_3', input=self.last_added_node)

			#Inception_layer 3
			nb_kernel_inception_3=[192,96,208,16,48,64]
			self.add_inception(self.last_added_node,nb_kernel_inception_3,'inception_3')

			#Inception_layer 4
			nb_kernel_inception_4=[160,112,224,24,64,64]
			self.add_inception(self.last_added_node,nb_kernel_inception_4,'inception_4')

			#Inception_layer 5
			nb_kernel_inception_5=[128,128,256,24,64,64]
			self.add_inception(self.last_added_node,nb_kernel_inception_5,'inception_5')

			#Inception_layer 6
			nb_kernel_inception_6=[112,144,288,32,64,64]
			self.add_inception(self.last_added_node,nb_kernel_inception_6,'inception_6')

			#Inception_layer 7
			nb_kernel_inception_7=[256,160,320,32,128,128]
			self.add_inception(self.last_added_node,nb_kernel_inception_7,'inception_7')

			pad_6 = self.add_to_graph(ZeroPadding2D((1,1)), name='pad_6', input=self.last_added_node)
			max_4 = self.add_to_graph(MaxPooling2D(pool_size=(3,3), strides=(2,2)), name='max_4', input=self.last_added_node)

			#Inception_layer 8
			nb_kernel_inception_8=[256,160,320,32,128,128]
			self.add_inception(self.last_added_node,nb_kernel_inception_8,'inception_8')

			#Inception_layer 9
			nb_kernel_inception_9=[384,192,384,48,128,128]
			self.add_inception(self.last_added_node,nb_kernel_inception_9,'inception_9')

			self.add_to_graph(AveragePooling2D((7,7)), name='pool_1', input=self.last_added_node)

			self.add_to_graph(Flatten(), name='flatten', input='pool_1')

			self.add_to_graph(Dense(self.cfgs['nb_classes']), name='logits', input='flatten')
			self.add_to_graph(Activation(self.cfgs['activation']), name='prob', input='logits')

			# Output to the Graph
			self.model.add_output(name='output', input='prob')

			self.init_from_this()

		except Exception as err:

			self.io.print_error('Error configuring the model, {0}'.format(err))
			self.init = False
			return

		#try

		self.init = True

	#def

	def init_from_this(self):

		weights_file = self.cfgs['model_weights_file']

		if not weights_file == 'None':
			self.load_weights(weights_file)
			self.io.print_info('Weights Initalized from {0}'.format(weights_file))
		#if

	#def

	def load_weights(self, filepath):

		if filepath.endswith('.npz'):
			pfile = open(filepath,'r')
			graph = np.load(pfile)['graph'].item()

			for node_name,weights in graph.items():

				if node_name in self.cfgs['ignore_while_loading']:
					self.io.print_warning('Ignoring weights from {0}'.format(node_name))
					continue
				#if

				self.io.print_info('Transfering parameters from {0}'.format(node_name))
				self.model.nodes[node_name].set_weights(weights)

			#for

			pfile.close()

		elif filepath.endswith('.hdf5'):
			self.model.load_weights(filepath)
		else:
			self.io.print_error('Unknown model weights file {}'.format(filepath))
		#if

		self.io.print_info(self.model.nodes['prob'].get_config())

	def setup_loss_function(self,w):

		self.loss_functions.set_weights(w)

	#def

	def compile(self,compile_cfgs):

		try:
			opt = optimizer_dict[compile_cfgs['optimizer']](lr=compile_cfgs['lr'], epsilon=compile_cfgs['epsilon'])
			self.model.compile(loss={'output':self.loss_functions.dict[compile_cfgs['loss']]}, optimizer=opt)
		except Exception as e:
			self.io.print_error('Error configuring the model, {0}'.format(e))
			self.init = False
			return
		#try

		self.init = True

	#def

#class
