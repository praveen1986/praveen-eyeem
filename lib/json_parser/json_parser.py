import os,sys
import json
import yaml
import numpy as np

class ConcereteParser():

	def __init__(self,config_file):

		if not os.path.exists(config_file):

			print 'Could not find config file, {0}'.format(config_file)
			self.init = False
			return 

		#if

		pfile = open(config_file,'r')
		self.cfgs = yaml.load(pfile)
		pfile.close()

		lemb_mapping_keys = [d[0] for d in self.cfgs['lemb_mapping_list']]
		lemb_mapping_values = [d[1] for d in self.cfgs['lemb_mapping_list']]

		self.lemb_mapping = dict(zip(lemb_mapping_keys,lemb_mapping_values))

		self.init = True

	#def

	def resolve_concepts(self,json_file):

		pfile = open(json_file,'r')
		json_d = json.load(pfile)
		pfile.close()

		resolved_concepts = self.resolve(json_d)

		return resolved_concepts

		#for

	#def

	def resolve(self,concepts):

		human_scores = concepts['concepts']['human_scores']

		# removes day/night conflict
		filtered_human_scores = self.resolve_antonyms(human_scores)

		# removes dangerous keywords
		filtered_concepts = self.resolve_stop_list(filtered_human_scores)

		# maps *-ethnicity to person
		mapped_concepts = self.map_lemb_concepts(filtered_concepts)

		return mapped_concepts

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

		  #print 'Supressing {0}:{1}'.format(human_labels[idx],human_scores[idx])

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

 	def resolve_stop_list(self,concepts):

		filtered_concepts = [ d[0] for d in concepts if d[0] not in self.cfgs['stop_list_lemb']]

		return filtered_concepts

	#def

	def map_lemb_concepts(self,concepts):

		return [ self.lemb_mapping[c] if c in self.lemb_mapping.keys() else c for c in concepts ]

	#def

#class

class AbstractParser():

	def __init__(self,config_file):

		if not os.path.exists(config_file):

			print 'Could not find asbtract config file, {0}'.format(config_file)
			self.init = False
			return

		#if

		pfile = open(config_file,'r')
		self.cfgs = yaml.load(pfile)
		pfile.close()

		self.init = True

		mapping_keys = [d[0] for d in self.cfgs['mapping_list']]
		mapping_values = [d[1] for d in self.cfgs['mapping_list']]

		self.mapping = dict(zip(mapping_keys,mapping_values))

	#def

	def resolve_concepts(self,json_file):

		pfile = open(json_file,'r')
		json_d = json.load(pfile)
		pfile.close()

		resolved_concepts = self.resolve(json_d)

		return resolved_concepts

	#def

	def resolve(self,concepts):


		top_abstract = concepts['concepts']['human']

		top_abstract = [ t for t in top_abstract if not t in self.cfgs['stop_list'] ]
		top_abstract = [ self.mapping[c] if c in self.mapping.keys() else c for c in top_abstract ]

		return top_abstract

	#def

#class

class AestheticsParser():

	def __init__(self,config_file):

		if not os.path.exists(config_file):

			print 'Could not find aesthetics config file, {0}'.format(config_file)
			self.init = False
			return 

		#if

		pfile = open(config_file,'r')
		self.cfgs = yaml.load(pfile)
		pfile.close()

		self.init = True

	#def

	def resolve_concepts(self,json_file):

		pfile = open(json_file,'r')
		json_d = json.load(pfile)
		pfile.close()

		resolved_concepts = self.resolve(json_d)

		return resolved_concepts

	#def

	def resolve(self,concepts):

		return concepts['concepts']['human']
		
	#def

#class