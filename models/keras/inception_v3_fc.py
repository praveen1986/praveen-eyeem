
# https://github.com/fchollet/keras/issues/1321#issuecomment-166576492

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

class InceptionV3_FC():

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
			self.io.print_error('Could not find config file for InceptionV3 {0}'.format(self.config_file))
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

		if 'padding' in kwargs:
			layer_name = layer_name + '_pad'
			self.add_to_graph(ZeroPadding2D(padding=kwargs['padding']), name=layer_name, input=input_layer)
			input_layer = layer_name
			del kwargs['padding']
		#if

		# CONV with linear activation by default
		layer_name = layer_name + '_conv'
		self.add_to_graph(Convolution2D(*args, border_mode='valid', **kwargs), name=layer_name, input=input_layer)

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

	def add_inceptionA(self, input_layer, list_nb_filter, base_name):

		l1_1 = self.add_bn_conv_layer(name=base_name+'_l1_1', input=input_layer, nb_filter=list_nb_filter[0][0], nb_row=1, nb_col=1)

		l2_1 = self.add_bn_conv_layer(name=base_name+'_l2_1', input=input_layer, nb_filter=list_nb_filter[1][0], nb_row=1, nb_col=1)
		l2_2 = self.add_bn_conv_layer(name=base_name+'_l2_2', input=l2_1, nb_filter=list_nb_filter[1][1], nb_row=5, nb_col=5, padding=(2,2))

		l3_1 = self.add_bn_conv_layer(name=base_name+'_l3_1', input=input_layer, nb_filter=list_nb_filter[2][0], nb_row=1, nb_col=1)
		l3_2 = self.add_bn_conv_layer(name=base_name+'_l3_2', input=l3_1, nb_filter=list_nb_filter[2][1], nb_row=3, nb_col=3, padding=(1,1))
		l3_3 = self.add_bn_conv_layer(name=base_name+'_l3_3', input=l3_2, nb_filter=list_nb_filter[2][2], nb_row=3, nb_col=3, padding=(1,1))

		l4_1 = self.add_to_graph(ZeroPadding2D(padding=(1,1)), name=base_name+'_14_1', input=input_layer)
		l4_2 = self.add_to_graph(AveragePooling2D(pool_size=(3,3), strides=(1,1)), name=base_name+'_14_2', input=l4_1)
		l4_3 = self.add_bn_conv_layer(name=base_name+'_l4_3', input=l4_2, nb_filter=list_nb_filter[3][0], nb_row=1, nb_col=1)

		self.add_to_graph(Activation("linear"), name=base_name, inputs=[l1_1, l2_2, l3_3, l4_3], merge_mode="concat", concat_axis=1)

		self.io.print_info('Added Inception-A {0}'.format(base_name))
		# https://github.com/fchollet/keras/issues/391

	def add_inceptionB(self, input_layer, list_nb_filter, base_name):

		l1_1 = self.add_bn_conv_layer(name=base_name+'_l1_1', input=input_layer, nb_filter=list_nb_filter[0][0], nb_row=3, nb_col=3, subsample=(2, 2))

		l2_1 = self.add_bn_conv_layer(name=base_name+'_l2_1', input=input_layer, nb_filter=list_nb_filter[1][0], nb_row=1, nb_col=1)
		l2_2 = self.add_bn_conv_layer(name=base_name+'_l2_2', input=l2_1, nb_filter=list_nb_filter[1][1], nb_row=3, nb_col=3, padding=(1,1))
		l2_3 = self.add_bn_conv_layer(name=base_name+'_l2_3', input=l2_2, nb_filter=list_nb_filter[1][2], nb_row=3, nb_col=3, subsample=(2,2))

		l3_1 = self.add_to_graph(MaxPooling2D(pool_size=(3,3), strides=(2,2)), name=base_name+'_13_1', input=input_layer)

		self.add_to_graph(Activation("linear"), name=base_name, inputs=[l1_1, l2_3, l3_1], merge_mode="concat", concat_axis=1)

		self.io.print_info('Added Inception-B {0}'.format(base_name))
		# https://github.com/fchollet/keras/issues/391

	def add_inceptionC(self, input_layer, list_nb_filter, base_name):

		l1_1 = self.add_bn_conv_layer(name=base_name+'_l1_1', input=input_layer, nb_filter=list_nb_filter[0][0], nb_row=1, nb_col=1)

		l2_1 = self.add_bn_conv_layer(name=base_name+'_l2_1', input=input_layer, nb_filter=list_nb_filter[1][0], nb_row=1, nb_col=1)
		l2_2 = self.add_bn_conv_layer(name=base_name+'_l2_2', input=l2_1, nb_filter=list_nb_filter[1][1], nb_row=1, nb_col=7, padding=(0,3))
		l2_3 = self.add_bn_conv_layer(name=base_name+'_l2_3', input=l2_2, nb_filter=list_nb_filter[1][2], nb_row=7, nb_col=1, padding=(3,0))
		## padding and nb_row might not match with the lasagne weights

		l3_1 = self.add_bn_conv_layer(name=base_name+'_l3_1', input=input_layer, nb_filter=list_nb_filter[2][0], nb_row=1, nb_col=1)
		l3_2 = self.add_bn_conv_layer(name=base_name+'_l3_2', input=l3_1, nb_filter=list_nb_filter[2][1], nb_row=7, nb_col=1, padding=(3,0))
		l3_3 = self.add_bn_conv_layer(name=base_name+'_l3_3', input=l3_2, nb_filter=list_nb_filter[2][2], nb_row=1, nb_col=7, padding=(0,3))
		l3_4 = self.add_bn_conv_layer(name=base_name+'_l3_4', input=l3_3, nb_filter=list_nb_filter[2][3], nb_row=7, nb_col=1, padding=(3,0))
		l3_5 = self.add_bn_conv_layer(name=base_name+'_l3_5', input=l3_4, nb_filter=list_nb_filter[2][4], nb_row=1, nb_col=7, padding=(0,3))

		l4_1 = self.add_to_graph(ZeroPadding2D(padding=(1,1)), name=base_name+'_14_1', input=input_layer)
		l4_2 = self.add_to_graph(AveragePooling2D(pool_size=(3,3), strides=(1,1)), name=base_name+'_14_2', input=l4_1)
		l4_3 = self.add_bn_conv_layer(name=base_name+'_l4_3', input=l4_2, nb_filter=list_nb_filter[3][0], nb_row=1, nb_col=1)

		self.add_to_graph(Activation("linear"), name=base_name, inputs=[l1_1, l2_3, l3_5, l4_3], merge_mode="concat", concat_axis=1)

		self.io.print_info('Added Inception-C {0}'.format(base_name))
		# https://github.com/fchollet/keras/issues/391

	def add_inceptionD(self, input_layer, list_nb_filter, base_name):

		l1_1 = self.add_bn_conv_layer(name=base_name+'_l1_1', input=input_layer, nb_filter=list_nb_filter[0][0], nb_row=1, nb_col=1)
		l1_2 = self.add_bn_conv_layer(name=base_name+'_l1_2', input=l1_1, nb_filter=list_nb_filter[0][1], nb_row=3, nb_col=3, subsample=(2,2))

		l2_1 = self.add_bn_conv_layer(name=base_name+'_l2_1', input=input_layer, nb_filter=list_nb_filter[1][0], nb_row=1, nb_col=1)
		l2_2 = self.add_bn_conv_layer(name=base_name+'_l2_2', input=l2_1, nb_filter=list_nb_filter[1][1], nb_row=1, nb_col=7, padding=(0,3))
		l2_3 = self.add_bn_conv_layer(name=base_name+'_l2_3', input=l2_2, nb_filter=list_nb_filter[1][2], nb_row=7, nb_col=1, padding=(3,0))
		l2_4 = self.add_bn_conv_layer(name=base_name+'_l2_4', input=l2_3, nb_filter=list_nb_filter[1][2], nb_row=3, nb_col=3, subsample=(2,2))

		l3_1 = self.add_to_graph(MaxPooling2D(pool_size=(3,3), strides=(2,2)), name=base_name+'_13_1', input=input_layer)

		self.add_to_graph(Activation("linear"), name=base_name, inputs=[l1_2, l2_4, l3_1], merge_mode="concat", concat_axis=1)

		self.io.print_info('Added Inception-D {0}'.format(base_name))
		# https://github.com/fchollet/keras/issues/391

	def add_inceptionE(self, input_layer, list_nb_filter, base_name, pool_mode):

		l1_1 = self.add_bn_conv_layer(name=base_name+'_l1_1', input=input_layer, nb_filter=list_nb_filter[0][0], nb_row=1, nb_col=1)

		l2_1 = self.add_bn_conv_layer(name=base_name+'_l2_1', input=input_layer, nb_filter=list_nb_filter[1][0], nb_row=1, nb_col=1)
		l2_2a = self.add_bn_conv_layer(name=base_name+'_l2_2a', input=l2_1, nb_filter=list_nb_filter[1][1], nb_row=1, nb_col=3, padding=(0,1))
		l2_2b = self.add_bn_conv_layer(name=base_name+'_l2_2b', input=l2_1, nb_filter=list_nb_filter[1][1], nb_row=3, nb_col=1, padding=(1,0))

		l3_1 = self.add_bn_conv_layer(name=base_name+'_l3_1', input=input_layer, nb_filter=list_nb_filter[2][0], nb_row=1, nb_col=1)
		l3_2 = self.add_bn_conv_layer(name=base_name+'_l3_2', input=l3_1, nb_filter=list_nb_filter[2][1], nb_row=3, nb_col=3, padding=(1,1))
		l3_3a = self.add_bn_conv_layer(name=base_name+'_l3_3a', input=l3_2, nb_filter=list_nb_filter[2][2], nb_row=1, nb_col=3, padding=(0,1))
		l3_3b = self.add_bn_conv_layer(name=base_name+'_l3_3b', input=l3_2, nb_filter=list_nb_filter[2][3], nb_row=3, nb_col=1, padding=(1,0))

		l4_1 = self.add_to_graph(ZeroPadding2D(padding=(1,1)), name=base_name+'_14_1', input=input_layer)
		l4_2 = self.add_to_graph(pooling_dict[pool_mode](pool_size=(3,3), strides=(1,1)), name=base_name+'_14_2', input=l4_1)
		l4_3 = self.add_bn_conv_layer(name=base_name+'_l4_3', input=l4_2, nb_filter=list_nb_filter[3][0], nb_row=1, nb_col=1)

		self.add_to_graph(Activation("linear"), name=base_name, inputs=[l1_1, l2_2a, l2_2b, l3_3a, l3_3b, l4_3], merge_mode="concat", concat_axis=1)

		self.io.print_info('Added Inception-E {0}'.format(base_name))
		# https://github.com/fchollet/keras/issues/391

	def define(self):

		try:

			self.model = Graph()
			#self.model.add_input(name='input',  input_shape=(self.cfgs['n_channels'], self.cfgs['image_height'], self.cfgs['image_width']))
			self.model.add_input(name='input',  input_shape=(self.cfgs['n_channels'], None, None))

			#
			# Part of the network which is defined in the config file should move here
			#

			cgfs_nodes = self.cfgs['nodes']

			for node in cgfs_nodes:

				if not node['type'] == 'Activation':
					self.add_to_graph(layer_dict[node['type']](**node['parameter']), name=node['name'] , input=node['input'])
				else:
					self.add_to_graph(layer_dict[node['type']](node['parameter']['mode']), name=node['name'] , input=node['input'])
				#if

				self.io.print_info('Added {1}:{0}'.format(node['type'],node['name']))

			#for

			self.add_inceptionA(input_layer=self.last_added_node, list_nb_filter=((64,), (48, 64), (64, 96, 96), (32,)), base_name='mixed')
			self.add_inceptionA(input_layer=self.last_added_node, list_nb_filter=((64,), (48, 64), (64, 96, 96), (64,)),  base_name='mixed_1')
			self.add_inceptionA(input_layer=self.last_added_node, list_nb_filter=((64,), (48, 64), (64, 96, 96), (64,)), base_name='mixed_2')

			self.add_inceptionB(input_layer=self.last_added_node, list_nb_filter=((384,), (64, 96, 96)), base_name='mixed_3')

			self.add_inceptionC(input_layer=self.last_added_node, list_nb_filter=((192,), (128, 128, 192), (128, 128, 128, 128, 192), (192,)), base_name='mixed_4')
			self.add_inceptionC(input_layer=self.last_added_node, list_nb_filter=((192,), (160, 160, 192), (160, 160, 160, 160, 192), (192,)), base_name='mixed_5')
			self.add_inceptionC(input_layer=self.last_added_node, list_nb_filter=((192,), (160, 160, 192), (160, 160, 160, 160, 192), (192,)), base_name='mixed_6')
			self.add_inceptionC(input_layer=self.last_added_node, list_nb_filter=((192,), (192, 192, 192), (192, 192, 192, 192, 192), (192,)), base_name='mixed_7')

			self.add_inceptionD(input_layer=self.last_added_node, list_nb_filter=((192, 320), (192, 192, 192, 192)), base_name='mixed_8')

			self.add_inceptionE(input_layer=self.last_added_node, list_nb_filter=((320,), (384, 384, 384), (448, 384, 384, 384), (192,)), pool_mode='average', base_name='mixed_9')
			self.add_inceptionE(input_layer=self.last_added_node, list_nb_filter=((320,), (384, 384, 384), (448, 384, 384, 384), (192,)), pool_mode='max', base_name='mixed_10')

			self.add_to_graph(AveragePooling2D(pool_size=(5,5)), name='pool3', input=self.last_added_node)
			self.io.print_info('Added {1}:{0}'.format('AveragePooling',self.last_added_node))

			"""
			self.add_to_graph(Flatten(), name='flatten', input='pool3')
			self.io.print_info('Added {1}:{0}'.format('Flatten',self.last_added_node))
			self.add_to_graph(Dense(self.cfgs['nb_classes']), name='softmax', input='flatten')
			self.io.print_info('Added {1}:{0}'.format('Dense',self.last_added_node))
			"""

			self.add_to_graph(Convolution2D(self.cfgs['nb_classes'],1,1),name='softmax',input='pool3')

			self.add_to_graph(Activation(self.cfgs['activation']), name='prob', input='softmax')

			# Output to the Graph
			self.model.add_output(name='output', input='prob')

			if not self.cfgs['model_weights_file'] == 'None':
				self.init_from_this()
			#if

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

		self.io.print_info(self.model.nodes['softmax'].get_config())

	#def

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
