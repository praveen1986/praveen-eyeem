import os,sys
import argparse
import time

from json_parser import ConcereteParser
from json_parser import AbstractParser
from json_parser import AestheticsParser

class MasterParser():

	def __init__(self,cfg_files_dict):

		self.init = self.check_paths([v for k,v in cfg_files_dict.items()])

		self.concrete_f = ConcereteParser(cfg_files_dict['concrete'])
		self.abstract_f = AbstractParser(cfg_files_dict['abstract'])
		self.aesthetics_f = AestheticsParser(cfg_files_dict['aesthetics'])

		if not self.init:
			print 'Could not find some of cfg YAML files'
			return
		#if

		self.all_parsers = {'concrete':self.concrete_f,'abstract':self.abstract_f,'aesthetics':self.aesthetics_f}		

		if not all([v.init for k,v in self.all_parsers.items()] ):
			print '[Error] Failed to configure, aborting'
		#if

	#def

	def time_it(self,json_files_dict,N):

		start = time.time()

		print 'Timing for {0} jsons'.format(N)

		for r in range(N):
			self.get_to_work(json_files_dict,verbose=False)
		#for

		done = time.time()

		elapsed = done - start

		print 'Done Timing {0} jsons in {1}'.format(N,elapsed)

	#def

	def get_to_work(self,json_files_dict,verbose=True):

		try:

			_concepts = []

			for key,json_file in json_files_dict.items():

				_concepts.extend(self.all_parsers[key].resolve_concepts(json_file))

				if verbose:
					print 'Done with {0}'.format(json_file)
				#if

			#for

			concepts = self.de_duplicate(_concepts)

			if verbose:
				print concepts
			#if

			return concepts

		except Exception as e:

			print 'Error parsing {0},{1}'.format(json_files_dict,e)

			return []

		#try

	#def

	def de_duplicate(self,some_list):

	    new_list = []

	    for i in some_list:
	        if i not in new_list:
	            new_list.append(i)
	        #if
	    #for

	    return new_list

	#def

	def check_paths(self,file_paths):

		return all([os.path.exists(f) for f in file_paths])

	#def

if __name__== '__main__':

	parser = argparse.ArgumentParser(description='Utility for parsing jsons and applying filtering rules')
	parser.add_argument('--run-filtering',dest='run_filterinig',action='store_true',default=False,help='Run processing of jsons')
	parser.add_argument('--run-timing',dest='time_it',action='store_true',default=False,help='Timing of json parsing')
	parser.add_argument('--N',dest='N',default=10000,help='Iterations over JSONs')
	parser.add_argument('--concrete-json',dest='concrete_json',help='Path to Concrete JSON file')
	parser.add_argument('--aesthetics-json',dest='aesthetics_json',help='Path to Aesthtics JSON file')
	parser.add_argument('--abstract-json',dest='abstract_json',help='Path to Abstract JSON file')
	parser.add_argument('--concrete-cfg-file',dest='concrete_cfg_file',help='Path to Concrete config file')
	parser.add_argument('--aesthetics-cfg-file',dest='aesthetics_cfg_file',help='Path to Aesthtics config file')
	parser.add_argument('--abstract-cfg-file',dest='abstract_cfg_file',help='Path to Abstract config file')
 
	args = parser.parse_args()

	# call once
	worker = MasterParser({'concrete':args.concrete_cfg_file,'abstract':args.abstract_cfg_file,'aesthetics':args.aesthetics_cfg_file})
	# 

	# class over a loop of triplets of jsons
	if args.run_filterinig and worker.init:
		human_concepts = worker.get_to_work({'concrete':args.concrete_json,'abstract':args.abstract_json,'aesthetics':args.aesthetics_json})
	#if

	if args.time_it and worker.init:
		worker.time_it({'concrete':args.concrete_json,'abstract':args.abstract_json,'aesthetics':args.aesthetics_json},args.N)
	#if


#if
