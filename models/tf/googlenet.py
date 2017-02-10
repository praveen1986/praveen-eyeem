import os
import yaml
import numpy as np
import tensorflow as tf

# the type of float to use throughout the session.
_FLOATX = 'float32'
_EPSILON = 10e-8

from eyelibs.lib.label_embedding.embedding_io import EmbeddingIO
from eyelibs.models.tf.network import Network
from eyelibs.lib.tf.optimizers import optimizer_dict
from eyelibs.lib.tf.loss_functions import loss_functions_dict

class GoogleNet(Network):

	def __init__(self):

		self.cfgs = None
		self.io = EmbeddingIO(None)
		self.init = False
		self.config_file = None

	#def

	def configure(self,config_file):

		self.config_file = config_file

		if not os.path.exists(self.config_file):
			self.io.print_info('Could not find the config file {}'.format(self.config_file))
			self.init = False
			return
		#if

		pfile = open(self.config_file,'r')
		self.cfgs = yaml.load(pfile)
		pfile.close()

		# tf Graph n_input
		self.images = tf.placeholder(tf.float32, shape=[None,self.cfgs['image_height'],self.cfgs['image_width'],self.cfgs['n_channels']])
		self.labels = tf.placeholder(tf.float32, shape=[None,self.cfgs['nb_classes']])
		self.keep_prob = tf.placeholder(tf.float32)

		Network.__init__(self,{'data':self.images})

	#def

	def define(self):

		# TODO : Add try catch 

		(self.feed('data')
			 .conv(7, 7, 64, 2, 2, name='conv1_7x7_s2')
			 .max_pool(3, 3, 2, 2, name='pool1_3x3_s2')
			 .lrn(2, 2e-05, 0.75, name='pool1_norm1')
			 .conv(1, 1, 64, 1, 1, name='conv2_3x3_reduce')
			 .conv(3, 3, 192, 1, 1, name='conv2_3x3')
			 .lrn(2, 2e-05, 0.75, name='conv2_norm2')
			 .max_pool(3, 3, 2, 2, name='pool2_3x3_s2')
			 .conv(1, 1, 64, 1, 1, name='inception_3a_1x1'))

		(self.feed('pool2_3x3_s2')
			 .conv(1, 1, 96, 1, 1, name='inception_3a_3x3_reduce')
			 .conv(3, 3, 128, 1, 1, name='inception_3a_3x3'))

		(self.feed('pool2_3x3_s2')
			 .conv(1, 1, 16, 1, 1, name='inception_3a_5x5_reduce')
			 .conv(5, 5, 32, 1, 1, name='inception_3a_5x5'))

		(self.feed('pool2_3x3_s2')
			 .max_pool(3, 3, 1, 1, name='inception_3a_pool')
			 .conv(1, 1, 32, 1, 1, name='inception_3a_pool_proj'))

		(self.feed('inception_3a_1x1', 
					 'inception_3a_3x3', 
					 'inception_3a_5x5', 
					 'inception_3a_pool_proj')
			 .concat(3, name='inception_3a_output')
			 .conv(1, 1, 128, 1, 1, name='inception_3b_1x1'))

		(self.feed('inception_3a_output')
			 .conv(1, 1, 128, 1, 1, name='inception_3b_3x3_reduce')
			 .conv(3, 3, 192, 1, 1, name='inception_3b_3x3'))

		(self.feed('inception_3a_output')
			 .conv(1, 1, 32, 1, 1, name='inception_3b_5x5_reduce')
			 .conv(5, 5, 96, 1, 1, name='inception_3b_5x5'))

		(self.feed('inception_3a_output')
			 .max_pool(3, 3, 1, 1, name='inception_3b_pool')
			 .conv(1, 1, 64, 1, 1, name='inception_3b_pool_proj'))

		(self.feed('inception_3b_1x1', 
					 'inception_3b_3x3', 
					 'inception_3b_5x5', 
					 'inception_3b_pool_proj')
			 .concat(3, name='inception_3b_output')
			 .max_pool(3, 3, 2, 2, name='pool3_3x3_s2')
			 .conv(1, 1, 192, 1, 1, name='inception_4a_1x1'))

		(self.feed('pool3_3x3_s2')
			 .conv(1, 1, 96, 1, 1, name='inception_4a_3x3_reduce')
			 .conv(3, 3, 208, 1, 1, name='inception_4a_3x3'))

		(self.feed('pool3_3x3_s2')
			 .conv(1, 1, 16, 1, 1, name='inception_4a_5x5_reduce')
			 .conv(5, 5, 48, 1, 1, name='inception_4a_5x5'))

		(self.feed('pool3_3x3_s2')
			 .max_pool(3, 3, 1, 1, name='inception_4a_pool')
			 .conv(1, 1, 64, 1, 1, name='inception_4a_pool_proj'))

		(self.feed('inception_4a_1x1', 
					 'inception_4a_3x3', 
					 'inception_4a_5x5', 
					 'inception_4a_pool_proj')
			 .concat(3, name='inception_4a_output')
			 .conv(1, 1, 160, 1, 1, name='inception_4b_1x1'))

		(self.feed('inception_4a_output')
			 .conv(1, 1, 112, 1, 1, name='inception_4b_3x3_reduce')
			 .conv(3, 3, 224, 1, 1, name='inception_4b_3x3'))

		(self.feed('inception_4a_output')
			 .conv(1, 1, 24, 1, 1, name='inception_4b_5x5_reduce')
			 .conv(5, 5, 64, 1, 1, name='inception_4b_5x5'))

		(self.feed('inception_4a_output')
			 .max_pool(3, 3, 1, 1, name='inception_4b_pool')
			 .conv(1, 1, 64, 1, 1, name='inception_4b_pool_proj'))

		(self.feed('inception_4b_1x1', 
					 'inception_4b_3x3', 
					 'inception_4b_5x5', 
					 'inception_4b_pool_proj')
			 .concat(3, name='inception_4b_output')
			 .conv(1, 1, 128, 1, 1, name='inception_4c_1x1'))

		(self.feed('inception_4b_output')
			 .conv(1, 1, 128, 1, 1, name='inception_4c_3x3_reduce')
			 .conv(3, 3, 256, 1, 1, name='inception_4c_3x3'))

		(self.feed('inception_4b_output')
			 .conv(1, 1, 24, 1, 1, name='inception_4c_5x5_reduce')
			 .conv(5, 5, 64, 1, 1, name='inception_4c_5x5'))

		(self.feed('inception_4b_output')
			 .max_pool(3, 3, 1, 1, name='inception_4c_pool')
			 .conv(1, 1, 64, 1, 1, name='inception_4c_pool_proj'))

		(self.feed('inception_4c_1x1', 
					 'inception_4c_3x3', 
					 'inception_4c_5x5', 
					 'inception_4c_pool_proj')
			 .concat(3, name='inception_4c_output')
			 .conv(1, 1, 112, 1, 1, name='inception_4d_1x1'))

		(self.feed('inception_4c_output')
			 .conv(1, 1, 144, 1, 1, name='inception_4d_3x3_reduce')
			 .conv(3, 3, 288, 1, 1, name='inception_4d_3x3'))

		(self.feed('inception_4c_output')
			 .conv(1, 1, 32, 1, 1, name='inception_4d_5x5_reduce')
			 .conv(5, 5, 64, 1, 1, name='inception_4d_5x5'))

		(self.feed('inception_4c_output')
			 .max_pool(3, 3, 1, 1, name='inception_4d_pool')
			 .conv(1, 1, 64, 1, 1, name='inception_4d_pool_proj'))

		(self.feed('inception_4d_1x1', 
					 'inception_4d_3x3', 
					 'inception_4d_5x5', 
					 'inception_4d_pool_proj')
			 .concat(3, name='inception_4d_output')
			 .conv(1, 1, 256, 1, 1, name='inception_4e_1x1'))

		(self.feed('inception_4d_output')
			 .conv(1, 1, 160, 1, 1, name='inception_4e_3x3_reduce')
			 .conv(3, 3, 320, 1, 1, name='inception_4e_3x3'))

		(self.feed('inception_4d_output')
			 .conv(1, 1, 32, 1, 1, name='inception_4e_5x5_reduce')
			 .conv(5, 5, 128, 1, 1, name='inception_4e_5x5'))

		(self.feed('inception_4d_output')
			 .max_pool(3, 3, 1, 1, name='inception_4e_pool')
			 .conv(1, 1, 128, 1, 1, name='inception_4e_pool_proj'))

		(self.feed('inception_4e_1x1', 
					 'inception_4e_3x3', 
					 'inception_4e_5x5', 
					 'inception_4e_pool_proj')
			 .concat(3, name='inception_4e_output')
			 .max_pool(3, 3, 2, 2, name='pool4_3x3_s2')
			 .conv(1, 1, 256, 1, 1, name='inception_5a_1x1'))

		(self.feed('pool4_3x3_s2')
			 .conv(1, 1, 160, 1, 1, name='inception_5a_3x3_reduce')
			 .conv(3, 3, 320, 1, 1, name='inception_5a_3x3'))

		(self.feed('pool4_3x3_s2')
			 .conv(1, 1, 32, 1, 1, name='inception_5a_5x5_reduce')
			 .conv(5, 5, 128, 1, 1, name='inception_5a_5x5'))

		(self.feed('pool4_3x3_s2')
			 .max_pool(3, 3, 1, 1, name='inception_5a_pool')
			 .conv(1, 1, 128, 1, 1, name='inception_5a_pool_proj'))

		(self.feed('inception_5a_1x1', 
					 'inception_5a_3x3', 
					 'inception_5a_5x5', 
					 'inception_5a_pool_proj')
			 .concat(3, name='inception_5a_output')
			 .conv(1, 1, 384, 1, 1, name='inception_5b_1x1'))

		(self.feed('inception_5a_output')
			 .conv(1, 1, 192, 1, 1, name='inception_5b_3x3_reduce')
			 .conv(3, 3, 384, 1, 1, name='inception_5b_3x3'))

		(self.feed('inception_5a_output')
			 .conv(1, 1, 48, 1, 1, name='inception_5b_5x5_reduce')
			 .conv(5, 5, 128, 1, 1, name='inception_5b_5x5'))

		(self.feed('inception_5a_output')
			 .max_pool(3, 3, 1, 1, name='inception_5b_pool')
			 .conv(1, 1, 128, 1, 1, name='inception_5b_pool_proj'))

		(self.feed('inception_5b_1x1', 
					 'inception_5b_3x3', 
					 'inception_5b_5x5', 
					 'inception_5b_pool_proj')
			 .concat(3, name='inception_5b_output')
			 .avg_pool(7, 7, 1, 1, padding='VALID', name='pool5_7x7_s1')
			 .dropout(self.keep_prob,name='dropout5')
			 .fc(self.cfgs['nb_classes'], relu=False, name='loss3_classifier')
			 .sigmoid(name='prob'))

		self.init = True

	#def

	def compile(self,compile_cfgs):

		# Constants dictating the learning rate schedule.
		RMSPROP_DECAY = 0.9                # Decay term for RMSProp.
		RMSPROP_MOMENTUM = 0.9             # Momentum in RMSProp.
		RMSPROP_EPSILON = 1.0              # Epsilon term for RMSProp.

		try:

			self.output = self.get_output(self.cfgs['optimization_node'])
			self.pred = self.get_output(self.cfgs['prediction_node'])

			# computation of performance metrics

			if compile_cfgs['metric'] == 'precision':
				self.t_pred = tf.cast(tf.greater(self.pred,compile_cfgs['global_threshold']),'float')
				self.correct_pred = tf.mul(self.t_pred,self.labels)
				self.accuracy = tf.reduce_sum(self.correct_pred)/(tf.reduce_sum(self.t_pred)+_EPSILON)
			elif compile_cfgs['metric'] == 'top-k accuracy':
				_, indices = tf.nn.top_k( self.pred, k=compile_cfgs['top_k'] )
				dense_top_k = tf.one_hot( tf.to_int64(indices), tf.shape(self.labels)[len(self.labels.get_shape()) - 1], 1.0, 0.0, axis = -1 )
				dense_top_k = tf.reshape(tf.reduce_sum( dense_top_k, 1, keep_dims=True ), tf.shape(self.pred))
				self.correct_pred = tf.mul( dense_top_k, self.labels )
				self.accuracy = tf.reduce_sum( self.correct_pred )/(tf.reduce_sum( dense_top_k )+_EPSILON)
			#if

			#
			# Make a distribution out of labels
			#

			self.loss = tf.reduce_mean(loss_functions_dict[compile_cfgs['loss']](self.output, self.labels))

			#
			# optimizer
			#
			global_step = tf.get_variable('global_step', [],initializer=tf.constant_initializer(0), trainable=False)

			# Calculate the learning rate schedule.
			decay_steps = compile_cfgs['stepsize']

			# Decay the learning rate exponentially based on the number of steps.
			lr = tf.train.exponential_decay(compile_cfgs['lr'],
																			global_step,
																			decay_steps,
																			compile_cfgs['decay'],
																			staircase=True)

			# Create an optimizer that performs gradient descent.
			opt = optimizer_dict[compile_cfgs['optimizer']](lr)


			self.optimizer = opt.minimize(self.loss)
			#
			# Initializing the variables
			init = tf.initialize_all_variables()

			self.sess = tf.Session()
			self.sess.run(init)

			self.load_weights()

			# good to go 
			self.init = True

		except Exception as err:

			self.io.print_error('Could not compile model {}'.format(err))
			self.init = False

		#try

	#def

	def load_weights(self):

		if not self.cfgs['model_weights_file'] == 'None':
			if self.cfgs['model_weights_file'].endswith('.npy'):

				data_dict = np.load(self.cfgs['model_weights_file']).item()

				for key in data_dict:

					with tf.variable_scope(key, reuse=True):
						for subkey, data in zip(('weights', 'biases'), data_dict[key]):

							if key in self.cfgs['ignore_while_loading']:
								self.io.print_warning('Skipping weights from {0}'.format(key))
								continue
							else:
								self.io.print_info('Porting weights from {0}'.format(key))
							#if

							try:
								var = tf.get_variable(subkey)
								self.sess.run(var.assign(data))
							except Exception as err:
								self.io.print_error('Error initializing layer {0}, {1}'.format(key,err))
							#try
						#for
					#with
				#for
				self.init = True

			elif 'snapshot' in self.cfgs['model_weights_file']:

				saver = tf.train.Saver()
				saver.restore(self.sess,self.cfgs['model_weights_file'])
				self.io.print_info('Model session restored from {0}'.format(self.cfgs['model_weights_file']))
				self.init = True

			else:
				self.io.print_info('File format for restoring not understood in {}'.format(self.cfgs['model_weights_file']))
				self.init = False
			#if

	#def