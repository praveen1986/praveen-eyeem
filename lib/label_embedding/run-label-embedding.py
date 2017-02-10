#!/usr/bin/env python

import os,sys
import argparse
from eyelibs.lib.label_embedding.label_embedding import LabelEmbedding
from termcolor import colored

def get_to_work(args):

	embedding_manager = LabelEmbedding(args.config_file,args.force)
	actions = {'run_preprocess':args.run_pre_process,'run_train':args.run_train,'build_embedding':args.build_embedding,'run_thresholds':args.run_thresholds,'run_test':args.run_test}
	embedding_manager.get_to_work(actions)

#def 

if __name__=='__main__':

	parser = argparse.ArgumentParser(description='Utility for setting-up Label embedding pre-processing/training/testing')
	parser.add_argument('--run-pre-process',dest='run_pre_process',action='store_true',default=False,help='Run pre-processing of training/validation data')
 	parser.add_argument('--run-train',dest='run_train',action='store_true',default=False,help='Launch training of Label embedding (Make sure you have pre-processed data before)')
 	parser.add_argument('--run-thresholds',dest='run_thresholds',action='store_true',default=False,help='Learn thresholds on the validation sets')
 	parser.add_argument('--build-embedding',dest='build_embedding',action='store_true',default=False,help='Build single embedding from individual embeddings')
 	parser.add_argument('--run-test',dest='run_test',action='store_true',default=False,help='Run test on test-set pointed to at in the config file')
 	parser.add_argument('--config-file',dest='config_file',type=str,required=True,default=None,help='YAML configuration file to set-up the experiment')
 	parser.add_argument('--resume',dest='resume',type=str,default=None,help='If to resume training from pre-existing snapshot')
 	parser.add_argument('--force',dest='force',action='store_true',default=False,help= 'Force specific tasks to over-write pre-existing files, else if not provided skips computation if files exist')
 
 	args = parser.parse_args()

 	if not ( args.run_pre_process or args.run_train or args.run_thresholds or args.run_test or args.build_embedding ):
 		print colored('[ERROR] Specify at-least one action, chose from --run-pre-process | --run-train | --build-embedding | --run_thresholds | --run-test  | --run-test-images ','red')
 		sys.exit(0)
 	#if

 	if os.path.exists(args.config_file):
		print colored('[INFO] Running Label Embedding with {0}'.format(args.config_file),'green')
		get_to_work(args)
 	else:
 		print colored('[ERROR] Could not find config file {0}, no such file'.format(args.config_file),'red')
 		sys.exit(0)
 	#else

#if