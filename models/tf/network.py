import numpy as np
import tensorflow as tf
from eyelibs.models.tf import scopes
from eyelibs.models.tf import variables
from tensorflow.python import control_flow_ops
from tensorflow.python.training import moving_averages

DEFAULT_PADDING = 'SAME'
# Collection containing all the variables created using slim.variables
MODEL_VARIABLES = '_model_variables_'
UPDATE_OPS_COLLECTION = '_update_ops_'

# Collection containing the slim.variables that are created with restore=True.
VARIABLES_TO_RESTORE = '_variables_to_restore_'

def layer(op):
	def layer_decorated(self, *args, **kwargs):
		# Automatically set a name if not provided.
		name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
		# Figure out the layer inputs.
		if len(self.inputs)==0:
			raise RuntimeError('No input variables found for layer %s.'%name)
		elif len(self.inputs)==1:
			layer_input = self.inputs[0]
		else:
			layer_input = list(self.inputs)
		# Perform the operation and get the output.
		layer_output = op(self, layer_input, *args, **kwargs)
		# Add to layer LUT.
		self.layers[name] = layer_output
		# This output is now the input for the next layer.
		self.feed(layer_output)
		# Return self for chained calls.
		return self
	return layer_decorated

class Network(object):

	def __init__(self, inputs, trainable=True):

		#
		# self.inputs holds all the layer
		#

		self.inputs = []
		self.layers = dict(inputs)
		self.trainable = trainable
		self.phase_train=tf.placeholder(tf.bool, name='phase_train')
		self.setup()

	def setup(self):
		pass

	# Legacy code not used anymore
	def load(self, data_path, session, ignore_missing=False):
		data_dict = np.load(data_path).item()
		for key in data_dict:
			with tf.variable_scope(key, reuse=True):
				for subkey, data in zip(('weights', 'biases'), data_dict[key]):
					try:
						var = tf.get_variable(subkey)
						session.run(var.assign(data))
					except ValueError:
						if not ignore_missing:
							raise
						#if
					#try
				#for
			#with
		#for
	#def

	def feed(self, *args):
		assert len(args)!=0
		self.inputs = []
		for layer in args:
			if isinstance(layer, basestring):
				try:
					layer = self.layers[layer]
				except KeyError:
					print self.layers.keys()
					raise KeyError('Unknown layer name fed: %s'%layer)
			self.inputs.append(layer)
		return self

	def get_output(self,node_name):

		return self.layers[node_name]

	def get_unique_name(self, prefix):
		id = sum(t.startswith(prefix) for t,_ in self.layers.items())+1
		return '%s_%d'%(prefix, id)
		
	def make_var(self, name, shape, initializer=None):
		return tf.get_variable(name, shape, trainable=self.trainable, initializer=initializer)

	def validate_padding(self, padding):
		assert padding in ('SAME', 'VALID')

	@layer
	def conv(self, input, k_h, k_w, c_o, s_h, s_w, name, relu=True, batch_norm=False, padding=DEFAULT_PADDING, group=1):

		self.validate_padding(padding)
		c_i = input.get_shape()[-1]

		assert c_i%group==0
		assert c_o%group==0

		convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)

		with tf.variable_scope(name) as scope:

			kernel = self.make_var('weights', shape=[k_h, k_w, c_i/group, c_o],initializer=tf.contrib.layers.xavier_initializer(uniform=False))
			biases = self.make_var('biases', shape=[c_o])

			if group==1:
				conv = convolve(input, kernel)
			else:
				input_groups = tf.split(3, group, input)
				kernel_groups = tf.split(3, group, kernel)
				output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
				conv = tf.concat(3, output_groups)
			#if

			
			if batch_norm:
				conv = self.batch_norm(conv)
			#if

			if relu:
				# [HSS] : To adapt to variable sized input making dimension invisible
				#bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
				bias = tf.nn.bias_add(conv, biases)
				return tf.nn.relu(bias, name=scope.name)
			#if

			# [HSS] : To adapt to variable sized input making dimension invisible
			#return tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list(), name=scope.name)

			return tf.nn.bias_add(conv, biases)

		#with

	#def

	@layer
	def relu(self, input, name):
		return tf.nn.relu(input, name=name)

	@layer
	def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
		self.validate_padding(padding)
		return tf.nn.max_pool(input,
								ksize=[1, k_h, k_w, 1],
								strides=[1, s_h, s_w, 1],
								padding=padding,
								name=name)

	@layer
	def avg_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
		self.validate_padding(padding)
		return tf.nn.avg_pool(input,
								ksize=[1, k_h, k_w, 1],
								strides=[1, s_h, s_w, 1],
								padding=padding,
								name=name)

	@layer
	def lrn(self, input, radius, alpha, beta, name, bias=1.0):
		return tf.nn.local_response_normalization(input,
													depth_radius=radius,
													alpha=alpha,
													beta=beta,
													bias=bias,
													name=name)

	@layer
	def concat(self, inputs, axis, name):
		return tf.concat(concat_dim=axis, values=inputs, name=name)

	@layer
	def fc(self, input, num_out, name, relu=True):
		with tf.variable_scope(name) as scope:
			input_shape = input.get_shape()
			if input_shape.ndims==4:
				dim = 1
				for d in input_shape[1:].as_list():
					dim *= d
				# [HSS] : To adapt to variable sized input making dimension invisible
				#feed_in = tf.reshape(input, [int(input_shape[0]), dim])
				feed_in = tf.reshape(input, [-1, dim])
			else:
				feed_in, dim = (input, int(input_shape[-1]))
			weights = self.make_var('weights', shape=[dim, num_out],initializer=tf.contrib.layers.xavier_initializer(uniform=False))
			biases = self.make_var('biases', shape=[num_out])
			op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
			fc = op(feed_in, weights, biases, name=scope.name)
			return fc

	@layer
	def softmax(self, input, name):
		return tf.nn.softmax(input, name)

	@layer
	def sigmoid(self,input,name):
		return tf.sigmoid(input,name)

	@layer
	def dropout(self, input, keep_prob, name):
		return tf.nn.dropout(input, keep_prob, name=name)

	@layer
	def flatten(self,input,name):
		return  tf.reshape(input, [-1, np.prod(input.get_shape()[1:].as_list())])

	def batch_norm(self,input,decay=0.9,epsilon=0.001,moving_vars='moving_vars',trainable=True,restore=True,scope=None):
		"""""
		[HSS] : Ported from Inception_v3 model
		Adds a Batch Normalization layer.
		Args:
			inputs: a tensor of size [batch_size, height, width, channels]
							or [batch_size, channels].
			decay: decay for the moving average.
			center: If True, subtract beta. If False, beta is not created and ignored.
			scale: If True, multiply by gamma. If False, gamma is
				not used. When the next layer is linear (also e.g. ReLU), this can be
				disabled since the scaling can be done by the next layer.
			epsilon: small float added to variance to avoid dividing by zero.
			moving_vars: collection to store the moving_mean and moving_variance.
			activation: activation function.
			is_training: whether or not the model is in training mode.
			trainable: whether or not the variables should be trainable or not.
			restore: whether or not the variables should be marked for restore.
			scope: Optional scope for variable_op_scope.
			reuse: whether or not the layer and its variables should be reused. To be
				able to reuse the layer scope must be given.
		Returns:
			a tensor representing the output of the operation.
		"""""

		inputs_shape = input.get_shape()

		with tf.variable_op_scope([input], scope, 'BatchNorm'):

			axis = list(range(len(inputs_shape) - 1))
			params_shape = inputs_shape[-1:]

			# Allocate parameters for the beta and gamma of the normalization.

			beta = variables.variable('beta',params_shape,initializer=tf.zeros_initializer,trainable=trainable,restore=restore)
			gamma = variables.variable('gamma',params_shape,initializer=tf.python.array_ops.ones,trainable=trainable,restore=restore)

			# Create moving_mean and moving_variance add them to
			# GraphKeys.MOVING_AVERAGE_VARIABLES collections.

			moving_collections = [moving_vars, tf.GraphKeys.MOVING_AVERAGE_VARIABLES]
			moving_mean = variables.variable('moving_mean',params_shape,initializer=tf.zeros_initializer,trainable=False,restore=restore,collections=moving_collections)
			moving_variance = variables.variable('moving_variance',params_shape,initializer=tf.python.array_ops.ones,trainable=False,restore=restore,collections=moving_collections)

			if self.phase_train == True:

				# Calculate the moments based on the individual batch.

				mean, variance = tf.nn.moments(input, axis)
				update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, decay)

				tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
				update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, decay)

				tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)

			else:

				# Just use the moving_mean and moving_variance.

				mean = moving_mean
				variance = moving_variance

			#if

			# Normalize the activations.

			outputs = tf.nn.batch_norm_with_global_normalization(input, mean, variance, beta, gamma, epsilon,True)
			outputs.set_shape(input.get_shape())

			return outputs

		#def