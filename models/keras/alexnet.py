# Alexnet in python
import os,sys
import yaml
import json

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

from rnd_libs.lib.label_embedding.embedding_io import EmbeddingIO

#AlexNet with batch normalization in Keras 
#input image is 224x224

class AlexNet():

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
			self.io.print_error('Could not find config file for Alexnet {0}'.format(self.config_file))
			self.init = False
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

		self.model = Sequential()
		self.model.add(Convolution2D(64, 3, 11, 11, border_mode=self.cfgs['border_mode']))
		self.model.add(BatchNormalization((64,226,226)))
		self.model.add(Activation(self.cfgs['non_linearity']))
		self.model.add(MaxPooling2D(poolsize=(3, 3)))

		self.model.add(Convolution2D(128, 64, 7, 7, border_mode=self.cfgs['border_mode']))
		self.model.add(BatchNormalization((128,115,115)))
		self.model.add(Activation(self.cfgs['non_linearity']))
		self.model.add(MaxPooling2D(poolsize=(3, 3)))

		self.model.add(Convolution2D(192, 128, 3, 3, border_mode=self.cfgs['border_mode']))
		self.model.add(BatchNormalization((128,112,112)))
		self.model.add(Activation(self.cfgs['non_linearity']))
		self.model.add(MaxPooling2D(poolsize=(3, 3)))

		self.model.add(Convolution2D(256, 192, 3, 3, border_mode=self.cfgs['border_mode']))
		self.model.add(BatchNormalization((128,108,108)))
		self.model.add(Activation(self.cfgs['non_linearity']))
		self.model.add(MaxPooling2D(poolsize=(3, 3)))

		self.model.add(Flatten())
		self.model.add(Dense(12*12*256, 4096, init=self.cfgs['fc_init']))
		self.model.add(BatchNormalization(4096))

		self.model.add(Activation(self.cfgs['non_linearity']))
		self.model.add(Dense(4096, 4096, init=self.cfgs['fc_init']))
		self.model.add(BatchNormalization(4096))

		self.model.add(Activation(self.cfgs['non_linearity']))
		self.model.add(Dense(4096, self.cfgs['nb_classes'], init=self.cfgs['fc_init']))
		self.model.add(BatchNormalization(self.cfgs['nb_classes']))

		self.model.add(Activation(self.cfgs['activation']))

	#def

	def compile():

		sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
		self.model.compile(loss=self.cfgs['loss'], optimizer=sgd)

	#def

	def train(self,X,Y):

		pass

	#def

	def fit(self,X,Y):

		pass

	#def

	def predict(self,X,Y):

		pass

	#def

	def predict_proba(X):

		pass

	#def

#class