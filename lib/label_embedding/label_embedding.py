#
# In this work here we will try learning a label embedding from dense predictions fully convolutions network to getty like tags
# Harsimrat Sandhawalia
# 15-06-2015
#

import os,sys
import numpy as np
import yaml
import json
from termcolor import colored
from random import shuffle
import cPickle
import sklearn
from rnd_libs.lib.label_embedding.embedding_io import EmbeddingIO

from sklearn.pipeline import Pipeline
from sklearn.externals import joblib

from rnd_libs.lib.label_embedding.classifiers import classifier_dict, normalizer_dict, scaling_dict

from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier

class LabelEmbedding(object):

	"""
		The main/useful class to implement some of the ideas in : https://hal.inria.fr/hal-00815747v1/document
		Given reasonable amout of data with dense predictions and getty tags we learn a mapping that can align them.
		arg_max( dense_predictions*W*getty_labels^T )
	"""

	def __init__(self,config_file,force=False,reduced=False):

		self.config_file = config_file
		self.init = True
		self.force = force

		self._read_configuration()
		self.init = self._verify_files()
		self.is_mat = True

		if not reduced:
			self.io = EmbeddingIO(self.cfgs['path_logging_dir'])
		else:
			self.io = EmbeddingIO(None,True,True)
			print 'Reduced functionality of Label Embedding'
		#if

		self.working_order = [(0,'run_preprocess',self._run_preprocess),(1,'run_train',self._run_training),(2,'build_embedding',self._build_embedding),(3,'run_thresholds',self._run_thresholds),(4,'run_test',self._run_testing)]
		self.train_method = {'CFM':self._solve_closed_form,'CFM_PC':self._solve_closed_form_pc,'SGD':self._solve_sgd,'WSABIE':self._solve_wsabie,'CCA':self._solve_cca}

		lemb_mapping_keys = [d[0] for d in self.cfgs['lemb_mapping_list']]
		lemb_mapping_values = [d[1] for d in self.cfgs['lemb_mapping_list']]

		self.lemb_mapping = dict(zip(lemb_mapping_keys,lemb_mapping_values))

	#def

	"""
		Check if all files and path exists.
		From yaml config file there are variables beginning with path_ 
	"""

	def prepare_for_testing(self):

		self._read_all_labels()

		if not '_PC' in self.cfgs['train_method']:
			self.W = self.io.read_weight_matrix(self.cfgs['npz_files_loc']['weights'])
			self.is_mat = True
			self.threshold_map, self.black_list = self.io.read_threshold_file(self.cfgs['npz_files_loc']['thresholds'])
			self.threshold_map = self.threshold_map.tolist()
			self.io.print_info('{0}/{1} concepts are black-listed (not enough validation data)'.format(len(self.black_list),len(self.getty_labels)))
		else:
			self.W = self.io.read_all_emb(self.getty_labels,self.cfgs['train_method'],self.cfgs['npz_files_loc']['weights'])
			self.is_mat = False
			if self.cfgs['use_global_threshold']:
				self.threshold_map, self.black_list = {c:0.5 for c in self.W.keys()},[]
			else:
				self.threshold_map, self.black_list = self.io.read_threshold_file(self.cfgs['npz_files_loc']['thresholds'])
			#if
			self.not_black_list = self.io.read_getty_labels(self.cfgs['path_getty_label_file'],self.cfgs['minimum_samples_for_validation'],self.cfgs['getty_key_words_types'])
		#if

		if self.is_mat:
			assert self.W.shape[1] == len(self.getty_labels)
			self.io.print_info('Found weight {0}x{1} matrix'.format(self.W.shape[0],self.W.shape[1]))
		#if

		

	def init_embedding(self,hedger_file,synset_names,getty_file,emb_weights_file,emb_threshold_file):

		self.all_wn_labels = self.io.read_hedged_labels(hedger_file)
		self.leaf_wn_labels = synset_names

		self.getty_labels = self.io.read_getty_labels(getty_file,self.cfgs['getty_stop_list'],self.cfgs['getty_key_words_types'])

		print('Reading label embedding from {0}'.format(emb_weights_file))

		if not '_PC' in self.cfgs['train_method']:
			self.W = self.io.read_weight_matrix(emb_weights_file)
			self.is_mat = True
			self.threshold_map, self.black_list = self.io.read_threshold_file(emb_threshold_file)
			self.threshold_map = self.threshold_map.tolist()
			self.io.print_info('{0}/{1} concepts are black-listed (not enough validation data)'.format(len(self.black_list),len(self.getty_labels)))
		else:

			self.W = self.io.read_all_emb(self.getty_labels,self.cfgs['train_method'],emb_weights_file)
			self.is_mat = False

			if self.cfgs['use_global_threshold']:
				self.threshold_map, self.black_list = {c:0.5 for c in self.W.keys()},[]
			else:
				self.threshold_map, self.black_list = self.io.read_threshold_file(self.cfgs['npz_files_loc']['thresholds'])
			#if

			self.not_black_list = self.io.read_getty_labels(getty_file,self.cfgs['minimum_samples_for_validation'],self.cfgs['getty_key_words_types'])
		#if

		if self.is_mat:
			assert self.W.shape[1] == len(self.getty_labels)
			self.io.print_info('Found weight {0}x{1} matrix'.format(self.W.shape[0],self.W.shape[1]))
		#if

		self.io.print_info('Testing set-up for label embedding')

	#if

	def _verify_files(self):

		flag = True

		paths_to_check = [ (k,v) for k,v in self.cfgs.items() if 'path_' in k ]

		for var,file_path in paths_to_check:

			if type(file_path) == dict:
				check_files = [ (var+'+'+r, j) for r,j in file_path.items() ]
			else:
				check_files = [ (var,file_path) ]
			#if

			file_flag = [ os.path.exists(f[1]) for f in check_files ]
			flag = flag and np.all(file_flag)

			for idx,f in enumerate(file_flag):
				if not f:
					print colored('[WARN] Could not find {0} for {1}, no such file or directory'.format(check_files[idx][1],check_files[idx][0]),'blue')
				#if
			#if

		#def

		return flag

	#def

	"""
		Simple yaml config file reader
	"""

	def _read_configuration(self):

		pfile = open(self.config_file)
		self.cfgs = yaml.load(pfile)
		pfile.close()

	#def

	"""
		Read wordnet synset file and getty labels. 
	"""

	def _read_all_labels(self):

		self.leaf_wn_labels, self.all_wn_labels = self.io.read_wn_labels(self.cfgs['path_bet_file'],self.cfgs['path_wn_labels_file'])
		self.getty_labels = self.io.read_getty_labels(self.cfgs['path_getty_label_file'],self.cfgs['getty_stop_list'],self.cfgs['getty_key_words_types'])
	#def

	def get_to_work(self,tasks):

		for idx, action, action_func in self.working_order:
			if tasks[action]:
				action_func()
			#if
		#for

	#def

	"""
		Read all possible labels wordnet and getty
		Run converting dense prediction vectors from DL into histogram features. 
		Run converting getty labels into binary vectors (presense or absence)
		Save processed vectors. 
	"""

	def _run_preprocess(self):

		self._read_all_labels()
		self._read_training_samples()
		self._process_vectors()

	#def

	"""
		Second step after pre-processing vectors, depending on training method selected used closed form solution or use SGD
	"""

	def _run_training(self):

			if self.cfgs['train_method'] in self.train_method.keys():
				self.train_method[self.cfgs['train_method']]()
			else:
				self.io.print_error('Could not find the specified training method {0}'.format(self.cfgs['train_method']))
			#if

	#def

	def _build_embedding(self):

		self._read_all_labels()

		W = self.io.read_all_emb(self.getty_labels,self.cfgs['train_method'],self.cfgs['path_weight_matrix'])

		self.io.write_all_emb(self.cfgs['npz_files_loc']['weights'],W)

		self.io.print_info('Done writing {1} embeddings to {0}'.format(self.cfgs['npz_files_loc']['weights'],len(W.keys())))

	#def

	"""
		Run testing woudl included and instance of FullConvNet and dense predictions.
		Mapping predictions using learned weight matrix.
		Save Mapped predictions. 
		TODO : Check code to reflect changes for individual embeddings
	"""

	def _run_thresholds(self):

		self.prepare_for_testing()

		self.validation_samples = self.io.read_images(self.cfgs['path_file_list']['val'])
		D_val, G_val,_ = self._read_all_data(self.validation_samples,'val')

		G_val_pred = [ v.predict_proba(D_val)[:,np.where(v.named_steps['train'].classes_==1)[0][0]] if type(v) == sklearn.pipeline.Pipeline else v.predict_proba(D_val)[:,np.where(v.classes_==1)[0][0]] for k,v in self.W.items()]

		G_val_pred = np.array(G_val_pred).transpose()
		
		sample_range = np.array(range(1,D_val.shape[0]+1),dtype=float)

		concepts_threshold = {}
		non_trained_concepts = []

		for l in range(D_val.shape[1]):

			gt_l = np.where(G_val[:,l]>0)[0].tolist()

			if len(gt_l) < self.cfgs['minimum_samples_for_validation']:
				self.io.print_warning('Skipping {0}, not enough samples {1}'.format(self.getty_labels[l],len(gt_l)))
				non_trained_concepts.append(self.getty_labels[l])
				continue
			#if

			sorted_idx = np.argsort(-G_val_pred[:,l])

			good_idx = [ 1 if idx in gt_l else 0 for idx in sorted_idx.tolist() ]
			c_sum = np.cumsum(np.array(good_idx))/sample_range

			here = np.where(c_sum>self.cfgs['concept_precision'])[0].tolist()
			fhere = here[-1] if not here == [] else np.argmax(c_sum)

			concepts_threshold[self.getty_labels[l]] = G_val_pred[sorted_idx[fhere],l]
			self.io.print_info('Concept {0} has threshold {1}'.format(self.getty_labels[l],concepts_threshold[self.getty_labels[l]]))

		#for

		self.io.save_threshold_file(self.cfgs['npz_files_loc']['thresholds'],concepts_threshold,non_trained_concepts)

	#def

	def predict(self,dense_pred):

		vec_dense = []

		for pred in dense_pred:
			vec_dense.extend(pred)
		#for

		predictions = self.dense_to_labels(vec_dense)
		predictions = [ (p) for p in predictions ]

		return predictions
		
	#def

	def resolve_label_embedding(self,emb_pred):

		filtered_pred = [ d[0] for d in emb_pred if d[0] not in self.cfgs['stop_list_lemb']]

		mapped_pred = self.map_lemb_concepts(filtered_pred)

		return mapped_pred

	#def

	def map_lemb_concepts(self,concepts):

		return [ self.lemb_mapping[c] if c in self.lemb_mapping.keys() else c for c in concepts ]

	#def

	def resolve_antonyms(self,human_predictions):

		conflicting_predictions = []
		human_dict = { d[0]:d[1] for d in human_predictions }

		human_labels = [ d[0] for d in human_predictions ]
		human_scores = [ d[1] for d in human_predictions ]

		for c1,c2 in self.cfgs['antonym_list']:
 
			if not ( c1 in human_labels and c2 in human_labels ):
				continue
			#if

			s1,s2 = human_dict[c1], human_dict[c2]
			idx = human_labels.index(c2) if s1 > s2 else human_labels.index(c1)

			print 'Supressing {0}:{1}'.format(human_labels[idx],human_scores[idx])

			del human_labels[idx]
			del human_scores[idx]

		#for

		remove_flag = -1

		for idx, group in enumerate(self.cfgs['count_order_list']):

			_this = np.intersect1d(human_dict.keys(),group)
				
			if len(_this) > 0:
				remove_flag = idx+1
				break
			#if

		#for

		if not remove_flag == len(self.cfgs['count_order_list']):

			remove_tags = []

			for g in self.cfgs['count_order_list'][remove_flag:]:
				remove_tags.extend(g)
			#for

			for t in remove_tags:

				if not t in human_labels:
					continue
				#if

				ridx = human_labels.index(t)

				del human_labels[ridx]
				del human_scores[ridx]

			#for

		#if

		return [ (g,s) for g,s in zip(human_labels,human_scores) ]

	#def

	def softmax(self,w, t = 1.0):

		e = np.exp(w / t)
		dist = e / np.sum(e)
		return dist

	#def

	def dense_to_labels(self,vec_dense):

		if vec_dense == []:
			return []
		#def 

		f_vec_dense = self._build_wn_label_vectors(vec_dense)

		if self.is_mat:
			getty_like_vec = np.dot(f_vec_dense,self.W)
			good = np.argsort(-getty_like_vec)
			good_labels = [ (self.getty_labels[g],getty_like_vec[g]) for g in good if ( not self.getty_labels[g] in self.black_list and self.getty_labels[g] in self.threshold_map.keys() ) ]
			good_labels = [ (g,s) for g,s in good_labels if s >= 0 ][:self.cfgs['top_k_tags']]
		else:
			getty_like_vec = [ (k,v.predict_proba(f_vec_dense)[0][np.where(v.named_steps['train'].classes_==1)[0][0]]) if type(v) == sklearn.pipeline.Pipeline else (k,v.predict_proba(f_vec_dense)[0][np.where(v.classes_==1)[0][0]]) for k,v in self.W.items()]
			good = np.argsort(-np.array([d[1] for d in getty_like_vec]))
			good_labels = [ getty_like_vec[g] for g in good if getty_like_vec[g][0] in self.not_black_list ]
			good_labels = [ (g,s) for g,s in good_labels if s > self.threshold_map[g] ][:self.cfgs['top_k_tags']]
		#if

		return good_labels

	#for

	def _run_testing(self):

		if not os.path.exists(self.cfgs['npz_files_loc']['weights']):
			self.io.print_error('Could not find label emdbedding matrix file {}'.format(self.cfgs['npz_files_loc']['weights']))
			return
		#if

		self.prepare_for_testing()
		self.testing_samples = self.io.read_images(self.cfgs['path_file_list']['test'])

		self.io.print_info('Will process {0} test images'.format(len(self.testing_samples)))

		shuffle(self.testing_samples)
		#self.testing_samples = self.testing_samples[110:120]

		for idx, sample in enumerate(self.testing_samples):

			if idx%1000 == 0:
				self.io.print_info('Processing {0}/{1} image'.format(idx,len(self.testing_samples)))
			#if

			basename = os.path.basename(sample)
			basename, ext = os.path.splitext(basename)

			json_file = os.path.join(self.cfgs['path_results_dir']['test'],basename+'.json')

			if os.path.exists(json_file):
				if not self.force:
					self.io.print_warning('Skipping {0}, already exists'.format(json_file))
					continue
				#if
			#if

			vec_dense = self.io.read_vector(os.path.join(self.cfgs['path_dense_pred_loc']['test'],basename)+self.cfgs['wn_suffix'])
			vec_getty = self.io.read_vector(os.path.join(self.cfgs['path_getty_labels_loc']['test'],basename)+self.cfgs['getty_suffix'])

			good_labels = self.dense_to_labels(vec_dense)

			self.io.save_good_labels(json_file,dict(good_labels))

			if self.cfgs['show_predicted_labels']:
				self.io.print_info('Photo-id {0}'.format(basename))
				self.io.print_info('Predicted {0}'.format(good_labels))
				self.io.print_warning('GT {0}'.format(vec_getty))
			#if
	
		#for

	#def

	"""
		Read images pointed to by training and validation file lists
	"""

	def _read_training_samples(self):

		self.training_data = {}
		self.validation_data = {}

		self.training_data['samples'] = self.io.read_images(self.cfgs['path_file_list']['train'])
		self.validation_data['samples'] = self.io.read_images(self.cfgs['path_file_list']['val'])

		self.io.print_info('Found {0} training samples {1} validation samples'.format(len(self.training_data['samples']),len(self.validation_data['samples'])))

	#def

	"""
		Convert dense predictions into vectors
		Convert getty labels into binary vectors
	"""

	def _process_vectors(self):

		self._label_to_vectors(self.training_data['samples'],'train')
		self._label_to_vectors(self.validation_data['samples'],'val')

	#def

	def _label_to_vectors(self,training_images,tag='train'):

		for idx,sample in enumerate(training_images):

			if idx % 1000 == 0:
				self.io.print_info('Processing {0}/{1} vectors for {2}, {3}'.format(idx,len(training_images),tag,sample))
			#if

			basename = os.path.basename(sample)
			basename, ext = os.path.splitext(basename)

			npz_file = os.path.join(self.cfgs['path_processed_vectors'][tag],basename+'.npz')

			if ( os.path.exists(npz_file) and not self.force ):
				self.io.print_info('Skipping {0}, already exists'.format(npz_file))
				continue
			#if

			vec_dense = self.io.read_vector(os.path.join(self.cfgs['path_dense_pred_loc']['train'],basename)+self.cfgs['wn_suffix'])
			vec_getty = self.io.read_vector(os.path.join(self.cfgs['path_getty_labels_loc']['train'],basename)+self.cfgs['getty_suffix'])

			#f_vec_dense = vec_dense

			f_vec_dense = self._build_wn_label_vectors(vec_dense)
			f_vec_getty = self._build_getty_label_vectors(vec_getty)

			self.io.save_npz_file(f_vec_dense,f_vec_getty,npz_file)

		#for

		self.io.print_info('Done building {0} vectors'.format(tag))

	#def
 
	"""
		Convert wn dense predictions to sparse vectors and L2 normalization. 
	"""

	def _build_wn_label_vectors(self,svec):

		if svec == []:
			return []
		#if

		vec = np.zeros(len(self.leaf_wn_labels),dtype=float)

		idx = np.array([ d[0] for d in svec ])
		values = np.array([ d[1] for d in svec ])

		for d in svec:
			vec[d[0]] += d[1]  
		#if

		if not self.cfgs['normalization']['wn_norm'] == 'None':
			vec /= (np.linalg.norm(vec)+0.000001)
		#if

		return vec

	#def

	"""
		Convert getty label vectors to binary vectors of presense of absence and L2 normalization. 
	"""

	def _build_getty_label_vectors(self,svec):

		if svec == []:
			 return []
		#if

		vec = np.zeros(len(self.getty_labels),dtype=float)
		idx = np.array([ self.getty_labels.index(d) for d in svec if d in self.getty_labels])

		if idx.size == 0:
		 return []
		#if

		vec[idx] = 1.0

		if not self.cfgs['normalization']['getty_norm'] == 'None':
			vec /= (np.linalg.norm(vec)+0.000001)
		#if

		return vec

	#def

	def _init_embedding_matrix(self):

		self.W = np.zeros((len(self.leaf_wn_labels),len(self.getty_labels)))
		self.io.print_info('Weight matrix set-up {0}x{1}'.format(self.W.shape[0],self.W.shape[1]))

	#def

	def _init_identity_matrix(self):

		self.I = np.eye(len(self.leaf_wn_labels),len(self.leaf_wn_labels))

	#def

	def _read_all_data(self,image_list,tag):

		D_all_vec = []
		G_all_vec = []
		all_files = []

		for idx,sample in enumerate(image_list):

			if idx % 1000 == 0:
				self.io.print_info('Reading {0}/{1} vectors for {3}, {2}'.format(idx,len(image_list),sample,tag))
			#if

			basename = os.path.basename(sample)
			basename, ext = os.path.splitext(basename)

			npz_file = os.path.join(self.cfgs['path_processed_vectors'][tag],basename+'.npz')

			if not os.path.exists(npz_file):
				self.io.print_warning('Skipping {0} no such file'.format(npz_file))
				continue
			#if

			D,G = self.io.read_npz_file(npz_file)

			D = D.flatten()

			if ( D.size == 0 or G.size == 0 or not len(D) == self.cfgs['expected_dim']):
				self.io.print_warning('Skipping {0} mis-match of dimensionality'.format(npz_file))
				continue
			#if

			D_all_vec.append(D)
			G_all_vec.append(G)
			all_files.append(basename)

		#for

		return np.array(D_all_vec), np.array(G_all_vec), all_files

	#def

	"""
		Different solutions to label embedding Closed form solution (CFM) / SGD / WSABIE / CCA
	"""

	def _solve_closed_form(self):

		self._read_all_labels()
		self._read_training_samples()

		#self._init_embedding_matrix()
		self._init_identity_matrix()

		D_train, G_train,_ = self._read_all_data(self.training_data['samples'],'train')

		if self.cfgs['balance_data']:
			D_train, G_train = self._balance_training_data(D_train,G_train)
		#if

		D_val, G_val,_ = self._read_all_data(self.validation_data['samples'],'val')

		train_size = (D_train.nbytes+G_train.nbytes)/(1000.0*1024**2)
		val_size = (D_val.nbytes+G_val.nbytes)/(1000.0*1024**2)

		# TODO : compute expected size of data in memory before reading it LOL
		# TODO : Future harsimrat will fix this
		# Yo future Harsimrat here, this is obsolte now. 

		self.io.print_info('Train : {0:2f} GB, val : {1:2f} GB'.format(train_size,val_size))

		for l in self.cfgs['lambda']:

			self.io.print_info('Computing closed form solution (CFM) for lambda = {0}'.format(l))
			self.W = np.dot(np.linalg.inv(np.dot(D_train.transpose(),D_train)+l*self.I),np.dot(D_train.transpose(),G_train))

			error = 1 - self._compute_validation_error_on_tags(self.W,D_val,G_val)

			self.io.print_info('(CFM) error for lambda = {0} : {1}'.format(l,error))
			self.io.save_weight_matrix(self.W,os.path.join(self.cfgs['path_weight_matrix'],'CFM-{0}.npz'.format(l)))

		#for

	#def

	"""
		Closed form solution per concepts
	"""

	def _solve_closed_form_pc(self):

		self._read_all_labels()
		self._read_training_samples()

		D_train, G_train,train_files = self._read_all_data(self.training_data['samples'],'train')
		D_val, G_val,val_files = self._read_all_data(self.validation_data['samples'],'val')

		train_size = (D_train.nbytes+G_train.nbytes)/(1000.0*1024**2)
		val_size = (D_val.nbytes+G_val.nbytes)/(1000.0*1024**2)

		self.io.print_info('Train : {0:2f} GB, val : {1:2f} GB'.format(train_size,val_size))
		self.reset_pipe_line()

		for label,label_name in len(self.getty_labels):

			svm_file_path = os.path.join(self.cfgs['path_weight_matrix'],'CFM_PC-{0}.pkl'.format(label_name))

			if os.path.exists(svm_file_path):
				if not self.force:
					self.io.print_warning('Skipping {}, already exists'.format(svm_file_path))
					continue
				#if
			#if

			tidx = np.sum(G_train[:,label])
			vidx = np.sum(G_val[:,label])

			if tidx == 0 or vidx == 0:
				self.io.print_warning('Not enough samples for {0}, train:{1} or val :{2}'.format(label_name,int(tidx),int(vidx)))
				continue
			#if

			self.io.print_info('{1}/{2} Learning embedding for {0} with {3} +samples {4} -ive samples'.format(label_name,label,len(self.getty_labels),np.sum(G_train[:,label]),G_train.shape[0]-np.sum(G_train[:,label])))

			self.pipe_line.fit(D_train,G_train[:,label])
			error = 1 - self.pipe_line.score(D_val,G_val[:,label])

			self.io.print_info('(CFM_PC) error for {1} : {0}'.format(error,label_name))

			self.io.save_emb_file(self.pipe_line,svm_file_path)

		#for

	#def

	def reset_pipe_line(self):

		action_list = [('norm',normalizer_dict[self.cfgs['normalization_type']]),('scale',scaling_dict[self.cfgs['scaling_type']]),('train',classifier_dict[self.cfgs['classifier_type']])]
		action_list = [(a,b) for a,b in action_list if not b == None]

		self.pipe_line = Pipeline(action_list)

	#def

	def _solve_sgd(self):

		self.io.print_error('SGD not implemented')

	#def

	def _solve_wsabie(self):

		self.io.print_error('WSABIE not implemented')

	#fef

	def _solve_cca(self):

		self.io.print_error('CCA not implemented')

	#def

	"""
		Validation needs to change since we optimize for tag coverage we should compute intersecton over union of predicted tags + getty tags on validation set
	"""

	def _compute_validation_error(self,W,D,G):

		n_samples = G.shape[0]
		pG = np.dot(D,W)

		for i in range(pG.shape[0]):
			pG[i,:] /= (np.linalg.norm(pG[i,:])+0.000001)
		#for

		cG = np.dot(pG,G.transpose())

		predictions = np.argmax(cG,axis=0)
		gt_match = np.array(range(n_samples))

		good = np.bitwise_xor(predictions,gt_match)
		correct_match = good == 0

		return np.sum(correct_match)/(n_samples+0.0)

	#def

	"""
		This function needs to change towards what we should optimize currently global predictions of tag is optimized. 

	"""

	def _compute_validation_error_on_tags(self,W,D,G):

		pG = np.dot(D,W)

		predictions = np.zeros(pG.shape)

		for i in range(pG.shape[0]):
			pidx = np.argsort(-pG[i,:])[:self.cfgs['top_k_tags']]
			predictions[i,pidx] = 1  
		#for

		gt = np.zeros(G.shape)
		idx = np.where(G>0)
		gt[idx] = 1

		good = predictions*gt
		n_samples = np.sum(predictions)

		#self.io.print_info('{0} correct out of {1}'.format(np.sum(good),n_samples))

		return np.sum(good)/(n_samples+0.0)

	#def

	def _compute_validation_error_on_tags_pc(self,W,D,G):

		pG = np.dot(D,W)
		predictions = np.zeros(pG.shape)

		pidx = np.where(pG>0)[0]
		predictions[pidx] = 1

		gt = G
		
		good = predictions*gt
		n_samples = np.sum(predictions)

		# error : (fp + tn)/n_samples

		return (np.sum(predictions) + np.sum(gt) - 2*np.sum(good))/(n_samples+0.0)

	#def

	def _restore_snapshot(self,label_idx):

		if label_idx == 0:
			self._init_embedding_matrix()
			return
		#if

		label = labe_idx -1
		weight_matrix_path = os.path.join(self.cfgs['path_weight_matrix'],'CFM-{1}.npz'.format(l,self.getty_labels[label]))

		self.W = self.io.read_weight_matrix(weight_matrix_path)

	#def

	def _balance_training_data(self,DL_train,G_train):




		return DL_train_balance, G_train_balance

	#def

#class