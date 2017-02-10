import os, sys

import argparse

from keras import backend as K

try:
    import keras.backend.tensorflow_backend as KTF
except ImportError as e:
    KTF = None
    TENSORFLOW_IMPORT_ERROR = e

from rnd_libs.lib.keras.train import TheanoTrainer
from rnd_libs.lib.keras.test import TheanoTester
from rnd_libs.lib.keras.train_captions import TheanoTrainerCaptions
from rnd_libs.lib.keras.test_captions import TheanoTesterCaptions
from rnd_libs.lib.keras.lmdb_parser import LMDBParser

trainer_dict = {'concepts': TheanoTrainer, 'captions': TheanoTrainerCaptions}
tester_dict = {'concepts': TheanoTester, 'captions': TheanoTesterCaptions}


def get_session(gpu_fraction):

    import tensorflow as tf

    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


def main(args):

    if K._BACKEND == 'tensorflow':
        if KTF is None:
            raise TENSORFLOW_IMPORT_ERROR
        KTF.set_session(get_session(args.gpu_limit))

    if not (args.run_train or args.run_test or args.run_train_threaded):
        print 'Set atleast one of the options --train | --test | --train-threaded'
        parser.print_help()
        return

    if not (args.concepts or args.captions):
        print 'Either --captions or --concepts should be provided'
        parser.print_help()
        return

    trainer_type = 'concepts' if args.concepts else 'captions'
    tester_type = 'concepts' if args.concepts else 'captions'
    #import ipdb; ipdb.set_trace()
    if args.run_train:
        
        trainer = trainer_dict[trainer_type](args.config_file, args.verbose)
        #import ipdb; ipdb.set_trace()
        trainer.setup()
        trainer.run()
        #trainer.run_validation(1)
    if args.run_train_threaded:

        trainer = trainer_dict[trainer_type](args.config_file, args.verbose)
        trainer.setup()
        trainer.run_threaded()

    if args.run_test:

        tester = tester_dict[trainer_type](args.config_file, args.verbose, args.raw_predictions)
        tester.setup()
        tester.run()
        tester.compare_various_betas()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Utility for Training/Testing DL models(Concepts/Captions) using theano/keras')
    parser.add_argument('--config-file', dest='config_file',type=str, help='Experiment configuration file')
    parser.add_argument('--train', dest='run_train', action='store_true', default=True, help='Launch training')
    parser.add_argument('--concepts', dest='concepts', action='store_true', default=True, help='Use concepts trainer')
    parser.add_argument('--captions', dest='captions', action='store_true', default=False, help='Use captions trainer')
    parser.add_argument('--train-threaded', dest='run_train_threaded', action='store_true', default=False, help='Launch threaded training')
    parser.add_argument('--test', dest='run_test', action='store_true', default=False, help='Launch testing on a list of images')
    parser.add_argument('--verbose', dest='verbose', default=0, type=int, help='Set verbosity level 0=minimal,3=a lot,4=biblical proportions')
    parser.add_argument('--gpu-limit', dest='gpu_limit', type=float, default=1.0, help='Use fraction of GPU memory (Useful with TensorFlow backend)')
    parser.add_argument('--raw-predictions', dest='raw_predictions', action='store_true', default=True, help='No application of heuristics and thresholds (Useful for keyword confidence measure)')
    
    args = parser.parse_args()

    main(args)
