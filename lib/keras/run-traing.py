import os,sys
import argparse

from rnd_libs.lib.keras_lib.train import TheanoTrainer
from rnd_libs.lib.keras_lib.test import TheanoTester
from rnd_libs.lib.keras_lib.lmdb_parser import LMDBParser

import theano

def main(args):

 	if not ( args.run_train or args.run_test or args.run_eval ):
 		print 'Set atleast one of the options --train | --test'
 		parser.print_help()
 		return
 	#if

 	if args.run_train:

 		trainer = TheanoTrainer(args.config_file,args.plot,args.verbose,args.send_updates)
 		trainer.setup()
 		trainer.run()

 	#if

 	if args.run_test or args.run_eval:

 		tester = TheanoTester(args.config_file,args.verbose)

 		if args.run_test:
 			tester.setup()
 			tester.run()
 		#if

 		if args.run_eval:
 			tester.eval()
 		#if

 	#if

#def

if __name__=='__main__':

	parser = argparse.ArgumentParser(description='Utility for Training/Testing DL models using theano/keras')
	parser.add_argument('--config-file',dest='config_file',type=str,help='Experiment configuration file')
 	parser.add_argument('--train',dest='run_train',action='store_true',default=False,help='Launch training')
 	parser.add_argument('--test',dest='run_test',action='store_true',default=False,help='Launch testing on a list of images')
 	parser.add_argument('--eval',dest='run_eval',action='store_true',default=False,help='Launch evaludation of the last test run')
 	parser.add_argument('--verbose',dest='verbose',default=0,type=int,help='Set verbosity level 0=minimal,3=a lot,4=biblical proportions')
 	parser.add_argument('--plot',dest='plot',action='store_true',default=False,help='Plot training testing progress')
 	parser.add_argument('--send-text-update',dest='send_updates',action='store_true',default=False,help='Send email updates on progress of training/testing')
 	
 	args = parser.parse_args()

 	main(args)

#if
