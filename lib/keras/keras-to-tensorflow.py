import numpy as np
import tensorflow as tf
import yaml
import argparse
from keras import backend as K
from rnd_libs.models.keras.models import model_dict
from termcolor import colored

def read_config_file(cfg_file):

	pfile = open(cfg_file,'r')
	cfgs = yaml.load(pfile)
	pfile.close()

	return cfgs

#def

def convert_kernel(kernel, dim_ordering='th'):

		'''Converts a kernel matrix (Numpy array)
		from Theano format to TensorFlow format
		(or reciprocally, since the transformation
		is its own inverse).
		'''
		new_kernel = np.copy(kernel)
		if dim_ordering == 'th':
				w = kernel.shape[2]
				h = kernel.shape[3]
				for i in range(w):
						for j in range(h):
								new_kernel[:, :, i, j] = kernel[:, :, w - i - 1, h - j - 1]
		elif dim_ordering == 'tf':
				w = kernel.shape[0]
				h = kernel.shape[1]
				for i in range(w):
						for j in range(h):
								new_kernel[i, j, :, :] = kernel[w - i - 1, h - j - 1, :, :]
		else:
				raise Exception('Invalid dim_ordering: ' + str(dim_ordering))

		#return new_kernel.transpose([2,3,1,0])
		return new_kernel

#def
	 
def predict_test_image(model):

	test_file = '/nas/datasets/aesthetics/images/v1/neutral/10574430.jpg'

	image = cv2.imread(test_file)

	image_data = cv2.resize(image,(299,299)).astype(np.float32)

	image_data = image_data[:,:,(2,1,0)]

	image_data -= 128.0

	image_data /= 128.0
	
	image_data = image_data.transpose(2,0,1)

	image_data = image_data[None,...]

	import pdb
	pdb.set_trace()

	predictions = model.predict({input:image_data})

	print predictions[np.argsort(-predictions)[:10]]

#def

def main(args):

	try:

		cfgs = read_config_file(args.config_file)

		keras_model = model_dict[cfgs['model_name']]()
		keras_model.configure(cfgs['model_config_file'])

		keras_model.define()

		init = tf.initialize_all_variables()
		K.get_session().run(init)

		keras_model.init_from_this()

		in_node = keras_model.model.get_input(train=False)
		out_node = keras_model.model.get_output(train=False)
		
		pred = K.function([in_node], [out_node])

		print colored('Read keras model','green')

		saver = tf.train.Saver()
		saver.save(K.get_session(),args.output)

		print colored('Saved TF model at {}, output node name : {}'.format(args.output, out_node.name),'green')

	except Exception as err:

		print colored('Error building TF model {}'.format(err),'red')

# def

if __name__=='__main__':

	parser = argparse.ArgumentParser(description='Utility for translating keras models to Tensorflow binary proto')
	parser.add_argument('--config-file',dest='config_file',type=str,required=True,help='Configuration file')
	parser.add_argument('--output',dest='output',required=True,type=str,help='Output model file path')
	
	args = parser.parse_args()

	main(args)

#if