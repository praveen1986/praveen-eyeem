from rnd_libs.lib.label_embedding.embedding_io import EmbeddingIO
from rnd_libs.lib.keras.lmdb_parser import LMDBParser
from rnd_libs.models.keras.models import model_dict
from keras import backend as K
import os, sys
import yaml
from time import time
import keras
import shutil
import numpy as np
from multiprocessing.pool import ThreadPool
from rnd_libs.lib.keras.loss_functions import _EPSILON
pool = ThreadPool(processes=1)


class LogHistory(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracy = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss').tolist())


class TheanoTrainerCaptions():

    def __init__(self, config_file, verbose):

        if not os.path.exists(config_file):
            print 'Error : could not find config file'.format(config_file)
            self.init = False
            return

        self.trainer_config_file = config_file
        self.read_config()

        self.io = EmbeddingIO(self.cfgs['log_dir'], 'training')

        self.trainer_name = self.cfgs['model_name']
        self.model_config_file = self.cfgs['model_config_file']

        self.verbose = verbose
        self.init = False

        if self.trainer_name not in model_dict.keys():
            self.io.print_error('Not a valid model type {0} chose among {1}'.format(self.trainer_name, model_dict.keys()))
            self.init = False
            return

        if not os.path.exists(self.trainer_config_file):
            self.io.print_error('Could not find configuration file {0}'.format(self.trainer_config_file))
            self.init = False
            return

        self.trainer = None
        self.train_lmdb = None
        self.val_lmdb = None
        self.all_labels = None

        self.save_configs()

    def save_configs(self):

        try:
            shutil.copy(self.trainer_config_file, self.cfgs['snapshot_dir'])
        except Exception as err:
            self.io.print_info('Using copy at destination already {}'.format(self.trainer_config_file))

        for var in self.cfgs['save_info']:
            try:
                value = self.cfgs[var]
                shutil.copy(value, self.cfgs['snapshot_dir'])
            except Exception as e:
                self.io.print_warning('Could not find requested attribute {0}'.format(var, e))

    def read_config(self):

        pfile = open(self.trainer_config_file)
        self.cfgs = yaml.load(pfile)
        pfile.close()

    def setup(self):

        self.trainer = model_dict[self.trainer_name]()
        self.trainer.configure(self.model_config_file)

        self.trainer.define()

        self.model_type = self.trainer.model.get_config()['name']

        self.im_h, self.im_w = self.trainer.cfgs['image_height'], self.trainer.cfgs['image_width']

        self.load_mean_file()
        self.load_synset_file()

        self.setup_training_data()
        self.setup_loss_function()

        if not self.trainer.init:
            self.io.print_error('Error with model definition')
            self.init = False
            return

        self.io.print_info('{0} Model defined from {1}'.format(self.trainer_name, self.model_config_file))
        self.trainer.compile(self.cfgs)
        self.compile_inference()

        if not self.trainer.init:
            self.io.print_error('Error with model complication')
            self.init = False
            return

        self.io.print_info('{0} Model compiled'.format(self.trainer_name))

        self.init = True

    def compile_inference(self):

        """
            This compile version is to be used for testing only. Since its optimized for predictions
        """

        if self.model_type == 'Sequential':
            self.inference = K.function(self.trainer.model.get_input(train=False), [self.trainer.model.layers[-1].get_output(train=False)])
        elif self.model_type == 'Graph':
            self.inference = K.function(self.trainer.model.get_input(train=False), [self.trainer.model.nodes[node].get_output(train=False) for node in self.cfgs['output_node_name']['test']])
        else:
            self.io.print_error('Say whaaaat ?! This model type is not supported : {}'.format(self.model_type))
            sel.init = False

        self.io.print_info('Inference model compiled')

    def setup_training_data(self):

        # TODO [HSS]: Reduce the argument lenght of set_params
        # This bit of set-up is not clean to read
        self.train_lmdb = LMDBParser(self.cfgs['train_file'], log_dir=self.cfgs['log_dir'])
        self.val_lmdb = None

        if 'val_file' in self.cfgs.keys():
            self.val_lmdb = LMDBParser(self.cfgs['val_file'], log_dir=self.cfgs['log_dir'])

        self.label_image_mapping = None

        if 'label_file' in self.cfgs.keys():
            self.label_image_mapping = self.io.read_npz_file(self.cfgs['label_file'], 'info').item()
            self.io.print_info('Read label/image lookup file {0}'.format(self.cfgs['label_file']))

        self.train_lmdb.set_params((self.im_h, self.im_w),
                                   self.mean_pixel_value,
                                   self.cfgs['train_lmdb_keep'],
                                   self.channel_swap,
                                   self.pixel_scaling,
                                   self.cfgs['label_sampling'],
                                   self.concepts,
                                   self.label_image_mapping,
                                   image_shuffle=self.image_shuffle)

        self.val_lmdb.set_params((self.im_h, self.im_w),
                                 self.mean_pixel_value,
                                 self.cfgs['val_lmdb_keep'],
                                 self.channel_swap,
                                 self.pixel_scaling,
                                 'None',
                                 self.concepts,
                                 None,
                                 image_shuffle=self.image_shuffle)

        if 'captions' in self.cfgs['model_name']:
            self.train_lmdb.set_captions_params(self.trainer.LUT,
                                                self.trainer.cfgs['max_caption_len'],
                                                self.trainer.vocabulary,
                                                self.cfgs['label_sampling'])
            self.val_lmdb.set_captions_params(self.trainer.LUT,
                                              self.trainer.cfgs['max_caption_len'],
                                              self.trainer.vocabulary,
                                              'captions')

    def setup_loss_function(self):

        self.trainer.setup_loss_function(self.train_lmdb.concepts_cdf)

    def load_mean_file(self):

        if not self.cfgs['mean_file'] == 'None':

            pfile = open(self.cfgs['mean_file'], 'r')
            m = np.load(pfile)
            pfile.close()

            m = np.mean(np.mean(m, axis=1), axis=1)
            self.mean_pixel_value = np.array([m[2], m[1], m[0]], dtype=np.float32)

        else:
            self.mean_pixel_value = np.array([128.0, 128.0, 128.0])

        self.pixel_scaling = self.cfgs['pixel_scaling']['train'] if 'pixel_scaling' in self.cfgs.keys() else None

        self.channel_swap = self.cfgs['channel_swap']['train'] if 'channel_swap' in self.cfgs.keys() else None

        self.image_shuffle = self.cfgs['image_shuffle']['train'] if 'image_shuffle' in self.cfgs.keys() else None

    def load_synset_file(self):

        pfile = open(self.cfgs['synset_file'], 'r')
        concepts = pfile.readlines()
        pfile.close()

        self.concepts = [p.strip() for p in concepts]

    def run(self):

        if not self.init:
            self.io.print_error('Error initializing trainer')
            return

        self.show_train_info()
        self.save_model_def()

        prefix = self.cfgs['snapshot_prefix']
        model_name = self.trainer.cfgs['model_weights_file']

        #
        # HSS : Shorter argument list ?
        #

        if prefix in model_name:
            start_iter = int(model_name.split(prefix)[-1].split('.')[0][1:])
            num_epocs = start_iter / self.cfgs['epoc_size']
            lr_rates = [self.trainer.update_optimizer(self.cfgs['decay']) for i in range(num_epocs)]
            self.io.print_info('Learning rate updated for snapshot {},{}'.format(model_name, lr_rates))
        else:
            start_iter = -1

        for n_iter in range(start_iter + 1, self.cfgs['maximum_iteration']):

            self.io.print_info('At iteration {0} of {1}'.format(n_iter, self.cfgs['maximum_iteration']))

            if n_iter % self.cfgs['stepsize'] == 0 and not n_iter == 0:
                self.update_lr()

            _, images, partial_captions, next_words = self.train_lmdb.get_batch[self.cfgs['label_sampling']]()

            train_logs = LogHistory()

            try:

                self.fit([images, partial_captions], next_words, batch_size=self.cfgs['batch_size'], nb_epoch=self.cfgs['nb_epocs'], callbacks=[train_logs], verbose=1)

            except Exception as e:

                self.io.print_info('Skipped iteration {0}, {1}'.format(n_iter, e))

            if n_iter % self.cfgs['display'] == 0 and not n_iter == 0:
                self.save_train_logs(n_iter, train_logs)

            if n_iter % self.cfgs['test_interval'] == 0 and not n_iter == 0:
                self.run_validation(n_iter)

            if n_iter % self.cfgs['snapshot'] == 0 and not n_iter == 0:
                self.save_snapshot(n_iter)

            if n_iter % self.cfgs['epoc_size'] == 0 and not n_iter == 0:

                last_lr = self.trainer.opt.get_config()['lr']
                self.trainer.update_optimizer(self.cfgs['decay'])
                current_lr = self.trainer.opt.get_config()['lr']

                self.io.print_info('[LR] update : last lr {}, current lr {}'.format(last_lr, current_lr))

        self.close_lmdbs()

    def run_threaded(self):

        if not self.init:
            self.io.print_error('Error initializing trainer')
            return

        self.show_train_info()

        self.save_model_def()

        prefix = self.cfgs['snapshot_prefix']
        model_name = self.trainer.cfgs['model_weights_file']

        #
        # HSS : Shorter argument list ?
        #

        if prefix in model_name:
            start_iter = int(model_name.split(prefix)[-1].split('.')[0][1:])
            num_epocs = start_iter / self.cfgs['epoc_size']
            lr_rates = [self.trainer.update_optimizer(self.cfgs['decay']) for i in range(num_epocs)]
            self.io.print_info('Learning rate updated for snapshot {},{}'.format(model_name, lr_rates))
        else:
            start_iter = -1

        _, images, partial_captions, next_words = self.train_lmdb.get_batch[self.cfgs['label_sampling']]()

        for n_iter in range(start_iter + 1, self.cfgs['maximum_iteration']):

            self.io.print_info('At iteration {0} of {1}'.format(n_iter, self.cfgs['maximum_iteration']))

            if n_iter % self.cfgs['stepsize'] == 0 and not n_iter == 0:
                self.update_lr()

            async_result = pool.apply_async(self.train_lmdb.get_batch[self.cfgs['label_sampling']])

            train_logs = LogHistory()

            try:

                t1 = time()
                self.fit([images, partial_captions], next_words, batch_size=self.cfgs['batch_size'], nb_epoch=self.cfgs['nb_epocs'], callbacks=[train_logs], verbose=1)
                t2 = time() - t1
                if self.cfgs['logging_for_profiling'] is True:
                    self.io.print_info('fit took {}s'.format(round(t2, 2)))
                t1 = time()
                self.eval(n_iter, [images, partial_captions], next_words)
                t2 = time() - t1
                if self.cfgs['logging_for_profiling']:
                    self.io.print_info('eval took {}s'.format(round(t2, 2)))

            except Exception as e:
                self.io.print_info('Skipped iteration {0}, {1}'.format(n_iter, e))

            if n_iter % self.cfgs['display'] == 0 and not n_iter == 0:
                self.save_train_logs(n_iter, train_logs)

            if n_iter % self.cfgs['test_interval'] == 0 and not n_iter == 0:
                self.run_validation(n_iter)

            if n_iter % self.cfgs['snapshot'] == 0 and not n_iter == 0:
                self.save_snapshot(n_iter)

            if n_iter % self.cfgs['epoc_size'] == 0 and not n_iter == 0:

                last_lr = self.trainer.opt.get_config()['lr']
                self.trainer.update_optimizer(self.cfgs['decay'])
                current_lr = self.trainer.opt.get_config()['lr']

                self.io.print_info('[LR] update : last lr {}, current lr {}'.format(last_lr, current_lr))

            t1 = time()
            _, images, partial_captions, next_words = async_result.get()

            t2 = time() - t1
            if self.cfgs['logging_for_profiling']:
                self.io.print_info('waiting for batch took {}s'.format(round(t2, 2)))

        self.close_lmdbs()

    def fit(self, input, output, **kwargs):

        if self.model_type == 'Sequential':
            self.trainer.model.fit(input, output, **kwargs)
        elif self.model_type == 'Graph':
            self.trainer.model.fit({'input': input, 'output': output}, **kwargs)
        else:
            self.io.print_error('Say whaaaat ?! This model type is not supported : {}'.format(self.model_type))

    def eval(self, n_iter, input, output, tag='TRAIN'):

        # NOTE : This is loss not accuracy

        pred = self.inference(input)[0]

        pred /= pred.sum(axis=-1, keepdims=True)

        # avoid numerical instability with _EPSILON clipping
        pred = np.clip(pred, _EPSILON, 1.0 - _EPSILON)

        if 'pn' in self.cfgs['loss']:
            val_loss = - np.sum(np.sum(output * np.log(pred) + (1 - output) * np.log(1 - pred), axis=-1), axis=-1)
        else:
            val_loss = - np.sum(np.sum(output * np.log(pred), axis=-1), axis=-1)

        self.io.print_info('{2} At {0} n_iter {1}'.format(n_iter, np.mean(val_loss), tag))

    def update_lr(self):

        pass

    def build_k_hot(self, svec, getty_dictionary):

        if svec == []:
            return []

        vec = np.zeros(len(getty_dictionary), dtype=float)
        idx = np.array([getty_dictionary.index(d) for d in svec if d in getty_dictionary])

        if idx.size == 0:
            return []

        vec[idx] = 1.0

        return vec

    def show_model_info(self):

        self.io.print_warning('BEGIN MODEL INFO')

        model = self.trainer.model

        if self.model_type == 'Sequential':

            for idx, layer in enumerate(model.layers):

                layer_info = layer.get_config()
                self.io.print_info('Layer {0} : {1}'.format(idx, layer_info))

        elif self.model_type == 'Graph':

            pass

        self.io.print_warning('END MODEL INFO')

    def show_train_info(self):

        self.io.print_warning('BEGIN TRAINING INFO')

        for var in self.cfgs['train_info']:
            try:
                value = self.cfgs[var]
                self.io.print_info('{0} : {1}'.format(var, value))
            except Exception as e:
                self.io.print_warning('Could not find requested attribute {0}'.format(var, e))

        self.io.print_warning('END TRAINING INFO')

    def save_model_def(self):

        yaml_string = self.trainer.model.to_yaml()
        model_definition_file = os.path.join(self.cfgs['snapshot_dir'], self.trainer_name + '_def.yaml')

        pfile = open(model_definition_file, 'w')
        pfile.write(yaml_string)
        pfile.close()

    def save_snapshot(self, n_iter):

        weights_file = os.path.join(self.cfgs['snapshot_dir'], '{0}_{2}_{1}.hdf5'.format(self.trainer_name, n_iter, self.cfgs['snapshot_prefix']))
        self.trainer.model.save_weights(weights_file, overwrite=True)
        self.io.print_info('Wrote snapshot to {0}'.format(weights_file))

    def run_validation(self, n_iter):

        self.io.print_info('Running validation')

        _, val_images, val_partial_captions, val_next_words = self.val_lmdb.get_batch['captions']()

        self.eval(n_iter, [val_images, val_partial_captions], val_next_words, tag='VALIDATION')

        self.io.print_info('End of validation')

    def save_train_logs(self, n_iter, train_logs):

        self.io.print_info('[{1}] TRAINING : loss:{0}'.format(np.mean(train_logs.losses), n_iter))

    def save_val_logs(self, n_iter, val_logs):

        self.io.print_info('[{1}] VALIDATION : loss:{0}'.format(val_logs, n_iter))

    def close_lmdbs(self):

        if self.train_lmdb is not None:
            self.train_lmdb.close()

        if self.val_lmdb is not None:
            self.val_lmdb.close()
