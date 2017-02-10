import sys,os
import yaml
import json

from keras.models import Graph, Sequential
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam

from rnd_libs.lib.label_embedding.embedding_io import EmbeddingIO
from rnd_libs.lib.keras_lib.loss_functions import loss_functions_dict

class SqueezeNet():

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
			self.io.print_error('Could not find config file for SqueezeNet {0}'.format(self.config_file))
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

			self.model.add_input(name='input',input_shape=(3,224,224))

			#conv 1
			self.model.add_node(Convolution2D(96, 3, 3, activation='relu', init='glorot_uniform',subsample=(2,2),border_mode='valid'),name='conv1', input='input')

			#maxpool 1
			self.model.add_node(MaxPooling2D((2,2)),name='pool1', input='conv1')

			#fire 2
			self.model.add_node(Convolution2D(16, 1, 1, activation='relu', init='glorot_uniform',border_mode='same'),name='fire2/conv1x1_1', input='pool1')
			self.model.add_node(Convolution2D(64, 1, 1, activation='relu', init='glorot_uniform',border_mode='same'),name='fire2/conv1x1_2', input='fire2/conv1x1_1')
			self.model.add_node(Convolution2D(64, 3, 3, activation='relu', init='glorot_uniform',border_mode='same'),name='fire2/conv3x3_2', input='fire2/conv1x1_1')
			self.model.add_node(Activation("linear"),name='fire2/concat', inputs=["fire2/conv1x1_2","fire2/conv3x3_2"], merge_mode="concat", concat_axis=1)

			#fire 3
			self.model.add_node(Convolution2D(16, 1, 1, activation='relu', init='glorot_uniform',border_mode='same'),name='"fire3/conv1x1_1', input='fire2/concat')
			self.model.add_node(Convolution2D(64, 1, 1, activation='relu', init='glorot_uniform',border_mode='same'),name='fire3/conv1x1_2', input='"fire3/conv1x1_1')
			self.model.add_node(Convolution2D(64, 3, 3, activation='relu', init='glorot_uniform',border_mode='same'),name='fire3/conv3x3_2', input='"fire3/conv1x1_1')
			self.model.add_node(Activation("linear"),name='fire3/concat', inputs=["fire3/conv1x1_2","fire3/conv3x3_2"], merge_mode="concat", concat_axis=1)

			#fire 4
			self.model.add_node(Convolution2D(32, 1, 1, activation='relu', init='glorot_uniform',border_mode='same'),name='"fire4/conv1x1_1', input='fire3/concat')
			self.model.add_node(Convolution2D(128, 1, 1, activation='relu', init='glorot_uniform',border_mode='same'),name='fire4/conv1x1_2', input='"fire4/conv1x1_1')
			self.model.add_node(Convolution2D(128, 3, 3, activation='relu', init='glorot_uniform',border_mode='same'),name='fire4/conv3x3_2', input='"fire4/conv1x1_1')
			self.model.add_node(Activation("linear"),name='fire4/concat', inputs=["fire4/conv1x1_2","fire4/conv3x3_2"], merge_mode="concat", concat_axis=1)

			#maxpool 4
			self.model.add_node(MaxPooling2D((2,2)),name='pool4', input='fire4/concat')

			#fire 5
			self.model.add_node(Convolution2D(32, 1, 1, activation='relu', init='glorot_uniform',border_mode='same'),name='fire5/conv1x1_1', input='pool4')
			self.model.add_node(Convolution2D(128, 1, 1, activation='relu', init='glorot_uniform',border_mode='same'),name='fire5/conv1x1_2', input='fire5/conv1x1_1')
			self.model.add_node(Convolution2D(128, 3, 3, activation='relu', init='glorot_uniform',border_mode='same'),name='fire5/conv3x3_2', input='fire5/conv1x1_1')
			self.model.add_node(Activation("linear"),name='fire5/concat', inputs=["fire5/conv1x1_2","fire5/conv3x3_2"], merge_mode="concat", concat_axis=1)

			#fire 6
			self.model.add_node(Convolution2D(48, 1, 1, activation='relu', init='glorot_uniform',border_mode='same'),name='fire6/conv1x1_1', input='fire5/concat')
			self.model.add_node(Convolution2D(192, 1, 1, activation='relu', init='glorot_uniform',border_mode='same'),name='fire6/conv1x1_2', input='fire6/conv1x1_1')
			self.model.add_node(Convolution2D(192, 3, 3, activation='relu', init='glorot_uniform',border_mode='same'),name='fire6/conv3x3_2', input='fire6/conv1x1_1')
			self.model.add_node(Activation("linear"),name='fire6/concat', inputs=["fire6/conv1x1_2","fire6/conv3x3_2"], merge_mode="concat", concat_axis=1)

			#fire 7
			self.model.add_node(Convolution2D(48, 1, 1, activation='relu', init='glorot_uniform',border_mode='same'),name='fire7/conv1x1_1', input='fire6/concat')
			self.model.add_node(Convolution2D(192, 1, 1, activation='relu', init='glorot_uniform',border_mode='same'),name='fire7/conv1x1_2', input='fire7/conv1x1_1')
			self.model.add_node(Convolution2D(192, 3, 3, activation='relu', init='glorot_uniform',border_mode='same'),name='fire7/conv3x3_2', input='fire7/conv1x1_1')
			self.model.add_node(Activation("linear"),name='fire7/concat', inputs=["fire7/conv1x1_2","fire7/conv3x3_2"], merge_mode="concat", concat_axis=1)

			#fire 8
			self.model.add_node(Convolution2D(64, 1, 1, activation='relu', init='glorot_uniform',border_mode='same'),name='fire8/conv1x1_1', input='fire7/concat')
			self.model.add_node(Convolution2D(256, 1, 1, activation='relu', init='glorot_uniform',border_mode='same'),name='fire8/conv1x1_2', input='fire8/conv1x1_1')
			self.model.add_node(Convolution2D(256, 3, 3, activation='relu', init='glorot_uniform',border_mode='same'),name='fire8/conv3x3_2', input='fire8/conv1x1_1')
			self.model.add_node(Activation("linear"),name='fire8/concat', inputs=["fire8/conv1x1_2","fire8/conv3x3_2"], merge_mode="concat", concat_axis=1)

			#maxpool 8
			self.model.add_node(MaxPooling2D((2,2)),name='pool8', input='fire8/concat')

			#fire 9
			self.model.add_node(Convolution2D(64, 1, 1, activation='relu', init='glorot_uniform',border_mode='same'),name='fire9/conv1x1_1', input='pool8')
			self.model.add_node(Convolution2D(256, 1, 1, activation='relu', init='glorot_uniform',border_mode='same'),name='fire9/conv1x1_2', input='fire9/conv1x1_1')
			self.model.add_node(Convolution2D(256, 3, 3, activation='relu', init='glorot_uniform',border_mode='same'),name='fire9/conv3x3_2', input='fire9/conv1x1_1')
			self.model.add_node(Activation("linear"),name='fire9/concat', inputs=["fire9/conv1x1_2","fire9/conv3x3_2"], merge_mode="concat", concat_axis=1)
			self.model.add_node(Dropout(0.5),input='fire9/concat',name='drop9')

			#conv 10
			self.model.add_node(Convolution2D(self.cfgs['nb_classes'], 1, 1, activation='relu', init='glorot_uniform',border_mode='valid'),name='conv_final', input='drop9')

			#avgpool 1
			self.model.add_node(AveragePooling2D((13,13)),name='pool_final', input='conv_final')

			self.model.add_node(Flatten(),name='flatten',input='pool_final')

			self.model.add_node(Activation(self.cfgs['activation']),input='flatten',name='prob')

			self.model.add_output(name='output',input='prob')

			if not self.cfgs['model_weights_file'] == None:
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
			self.io.print_info('Initalized from {0}'.format(weights_file))
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
			self.io.print_info('Transfering parameters from {0}'.format(layer_name_string))

			weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
			self.model.layers[k].set_weights(weights)

		#for

		f.close()

		self.io.print_info(self.model.layers[-1].get_config().values())

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