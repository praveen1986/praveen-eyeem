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
import cStringIO
from PIL import Image
import urlparse, urllib, StringIO
import cv2
pool = ThreadPool(processes=1)


class LogHistory(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracy = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss').tolist())


class TheanoTrainer():

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
    def scale_image(self, image):

            #
            # caffe scales image 0-1 in pixel values we bring it back to 255 as keras likes it that way
            #

            # No more caffe no more scaling down to 0-1 in caffe
            # image *= 255.0

            #tstart = time.time()

            if self.cfgs['square_crop']:
                if self.verbose:
                    self.io.print_info('Yo ! As you like it - Squaring up the image')
                image = self.square_it_up(image)

            # tend = time.time()

            # tstart = time.time()

            try:
                if self.cfgs['resize_images']:
                    image = cv2.resize(image, (self.im_h, self.im_w)).astype(np.float32)
                else:
                    print 'no resize'
                    image=image.astype(np.float32)
            except Exception as err:
                self.io.print_error('Could not parse {0}'.format(e))
                return None

            # tend = time.time()

            # tstart = time.time()

            if self.mean_pixel_value is not None:
                image -= self.mean_pixel_value

            if self.pixel_scaling is not None:
                image /= self.pixel_scaling

            if self.channel_swap is not None:
                image = image[:, :, self.channel_swap]

            # H,W,C to C,H,W
            if self.image_shuffle is not None:
                image = image.transpose(self.image_shuffle)

            #tend = time.time()

            return image
    def fetch_images(self, image_file_names):

        images = []
        basenames = []

        #tstart = time.time()

        for idx, image_file_name in enumerate(image_file_names):

            image_file_name=image_file_name.replace('/nas/','/nas2/')
            im_basename = os.path.basename(image_file_name)
            im_basename, _ = os.path.splitext(im_basename)

            if not urlparse.urlparse(image_file_name).scheme == "":

                url_response = urllib.urlopen(image_file_name)

                if url_response.code == 404:
                    print self.io.print_error('[Training] URL error code : {1} for {0}'.format(image_file_name, url_response.code))
                    continue

                try:

                    string_buffer = StringIO.StringIO(url_response.read())
                    image = np.asarray(bytearray(string_buffer.read()), dtype="uint8")

                    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                    image = image[:, :, (2, 1, 0)]
                    image = image.astype(np.float32)

                except Exception as err:

                    print self.io.print_error('[Training] Error with URL {0} {1}'.format(image_file_name, err))
                    continue

            elif os.path.exists(image_file_name):

                try:

                    fid = open(image_file_name, 'r')
                    stream = fid.read()
                    fid.close()

                    image = Image.open(cStringIO.StringIO(stream))
                    image = np.array(image)
                    image = image.astype(np.float32)

                    # image = cv2.imread(image_file_name)
                    # image = image[:, :, (2, 1, 0)]
                    # image = image.astype(np.float32)

                    if image.ndim == 2:
                        image = image[:, :, np.newaxis]
                        image = np.tile(image, (1, 1, 3))
                    elif image.shape[2] == 4:
                        image = image[:, :, :3]

                except Exception as err:

                    print self.io.print_error('[Training] Error with image file {0} {1}'.format(image_file_name, err))
                    continue
            else:

                try:

                    image = self.conn.read_image(image_file_name)

                    if image is None:
                        raise self.io.print_error("[Training] Image data could not be loaded - %s" % image_file_name)
                        continue

                    image = np.array(image)

                    if image.ndim == 2:
                        image = image[:, :, np.newaxis]
                        image = np.tile(image, (1, 1, 3))
                    elif image.shape[2] == 4:
                        image = image[:, :, :3]

                except Exception as err:

                    print self.io.print_error('[Training] Error with S3 Bucket hash {0} {1}'.format(image_file_name, err))
                    continue

            # try : do or do not there is not try
            #orig_image=image.astype('u1')
            image = self.scale_image(image)

            if image is None:
                continue

            if self.verbose:
                self.io.print_info('Processing {0} {1}'.format(image_file_name, image.shape))

            images.append(image)

            basenames.append(im_basename)

        #tend = time.time()

        #self.io.print_info('Fetching {0} images took {1:.4f} secs'.format(len(image_file_names), tend - tstart))
        if self.cfgs['resize_images']:
            return np.array(images)
        else:
            return images
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
    def see_model(self):
        
        self.trainer = model_dict[self.trainer_name]()
        self.trainer.configure(self.model_config_file)

        self.trainer.define()

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
            self.inference = K.function([self.trainer.model.get_input(train=False)], [self.trainer.model.layers[-1].get_output(train=False)])
        elif self.model_type == 'Graph':
            self.inference = K.function([self.trainer.model.get_input(train=False)], [self.trainer.model.nodes[node].get_output(train=False) for node in self.cfgs['output_node_name']['test']])
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
        self.label_embedding = None
        self.label_vectors = None

        # if 'label_file' in self.cfgs.keys():
        #     import ipdb; ipdb.set_trace()
        #     self.label_image_mapping = self.io.read_npz_file(self.cfgs['label_file'], 'info').item()
        #     #self.label_image_mapping =None
        #     self.io.print_info('Read label/image lookup file {0}'.format(self.cfgs['label_file']))

        if 'label_file' in self.cfgs.keys():

            images, mapping, vocab, vocab_mapping = self.io.read_npz_file(self.cfgs['label_file'], ['images', 'mapping', 'vocabulary', 'vocabulary_mapping'])

            self.label_image_mapping = {'images': list(images), 'mapping': list(mapping), 'vocabulary': list(vocab), 'vocabulary_mapping': vocab_mapping.item()}
            self.io.print_info('Read label/image lookup file {0}'.format(self.cfgs['label_file']))

        if 'label_embedding_file' in self.cfgs.keys():
            self.label_embedding = self.io.read_npz_file(self.cfgs['label_embedding_file'], 'matrix')
            self.label_vectors = self.io.read_npz_file(self.cfgs['label_embedding_file'], 'vectors')

            self.io.print_info('Read label embedding file {0}'.format(self.cfgs['label_embedding_file']))

        self.train_lmdb.set_params((self.im_h, self.im_w),
                                   self.mean_pixel_value,
                                   self.cfgs['train_lmdb_keep'],
                                   self.channel_swap,
                                   self.pixel_scaling,
                                   self.cfgs['label_sampling'],
                                   self.concepts,
                                   self.label_image_mapping,
                                   image_shuffle=self.image_shuffle,
                                   label_embedding=self.label_embedding)

        self.val_lmdb.set_params((self.im_h, self.im_w),
                                 self.mean_pixel_value,
                                 self.cfgs['val_lmdb_keep'],
                                 self.channel_swap,
                                 self.pixel_scaling,
                                 'None',
                                 self.concepts,
                                 None,
                                 image_shuffle=self.image_shuffle,
                                 label_embedding=self.label_embedding)

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
        #import ipdb; ipdb.set_trace()
        for n_iter in range(start_iter + 1, self.cfgs['maximum_iteration']):

            self.io.print_info('At iteration {0} of {1}'.format(n_iter, self.cfgs['maximum_iteration']))

            if n_iter % self.cfgs['stepsize'] == 0 and not n_iter == 0:
                self.update_lr()

            file_names, images, labels_train, labels_test = self.train_lmdb.get_batch[self.cfgs['label_sampling']]()
            
            orig_images=self.fetch_images(file_names)

            train_logs = LogHistory()

            try:
                # if self.cfgs['resize_images'] == False:
                #     for img,lab_train in zip(orig_images,labels_train):
                #         self.fit([img], [lab_train], batch_size=self.cfgs['batch_size'], nb_epoch=self.cfgs['nb_epocs'], callbacks=[train_logs], verbose=1)

                # else:
                
                self.fit(orig_images, labels_train, batch_size=self.cfgs['batch_size'], nb_epoch=self.cfgs['nb_epocs'], callbacks=[train_logs], verbose=1)

                self.eval(n_iter, orig_images, labels_test)

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

        _, images, labels_train, labels_test = self.train_lmdb.get_batch[self.cfgs['label_sampling']]()

        for n_iter in range(start_iter + 1, self.cfgs['maximum_iteration']):

            self.io.print_info('At iteration {0} of {1}'.format(n_iter, self.cfgs['maximum_iteration']))

            if n_iter % self.cfgs['stepsize'] == 0 and not n_iter == 0:
                self.update_lr()

            async_result = pool.apply_async(self.train_lmdb.get_batch[self.cfgs['label_sampling']])

            train_logs = LogHistory()

            try:

                t1 = time()
                self.fit(images, labels_train, batch_size=self.cfgs['batch_size'], nb_epoch=self.cfgs['nb_epocs'], callbacks=[train_logs], verbose=1)
                t2 = time() - t1
                if self.cfgs['logging_for_profiling'] is True:
                    self.io.print_info('fit took {}s'.format(round(t2, 2)))
                t1 = time()
                self.eval(n_iter, images, labels_test)
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
            _, images, labels_train, labels_test = async_result.get()

            t2 = time() - t1
            if self.cfgs['logging_for_profiling']:
                self.io.print_info('waiting for batch took {}s'.format(round(t2, 2)))

        self.close_lmdbs()

    def fit(self, images, labels, **kwargs):

        if self.model_type == 'Sequential':
            self.trainer.model.fit(images, labels, **kwargs)
        elif self.model_type == 'Graph':
            self.trainer.model.fit({'input': images, 'output': labels}, **kwargs)
        else:
            self.io.print_error('Say whaaaat ?! This model type is not supported : {}'.format(self.model_type))

    def eval(self, n_iter, images, labels, tag='TRAIN'):

        img_batches = [images[i:i + self.cfgs['batch_size']] for i in range(0, len(images), self.cfgs['batch_size'])]
        img_labels = [labels[i:i + self.cfgs['batch_size']] for i in range(0, len(labels), self.cfgs['batch_size'])]

        acc = 0.0

        for img, lab in zip(img_batches, img_labels):

            (pred, feat) = self.inference([img])

            """
            if self.model_type == 'Sequential':
                pred = self.trainer.model.predict(img)
            elif self.model_type == 'Graph':
                pred = self.trainer.model.predict({'input':img})[self.cfgs['output_node_name']['train']]
            else:
                self.io.print_error('Say whaaaat ?! This model type is not supported : {}'.format(self.model_type))
            #if
            """

            acc += self.compute_accuracy(pred, lab)

        acc /= len(img_batches)

        self.io.print_info('{3} At {0} n_iter {2} {1}'.format(n_iter, acc, self.cfgs['metric'], tag))

    def compute_accuracy(self, predictions, gt):

        if self.label_vectors is not None:
            # Need for L2 normalization of predictions and if need self.label_vectors
            predictions /= np.linalg.norm(predictions, axis=1)[:, np.newaxis]
            predictions = np.dot(predictions, self.label_vectors)

        pred_labels = np.zeros(predictions.shape)

        if self.cfgs['metric'] in ['precision']:

            pred_labels[np.where(predictions > self.cfgs['global_threshold'])] = 1.0

        elif self.cfgs['metric'] in ['top-k accuracy']:

            good_labels = np.argsort(-predictions, axis=1)[:, :self.cfgs['top_k']['train']]

            for idx in range(pred_labels.shape[0]):
                pred_labels[idx, good_labels[idx]] = 1

        else:

            self.io.print_warning('Accuracy not set for {0}'.format(self.cfgs['loss']))
            return 0.0

        g = pred_labels * gt + 0.0
        acc = np.nan_to_num(g.sum(axis=1) / (pred_labels.sum(axis=1))).mean()

        return acc

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

        keys, val_images, _, val_labels = self.val_lmdb.get_batch['None']()

        """
        if self.model_type == 'Sequential':
            val_logs = self.trainer.model.evaluate(val_images, val_labels, batch_size=1)
        elif self.model_type == 'Graph':
            val_logs = self.trainer.model.evaluate({'input':val_images,self.cfgs['output_node_name']['train']:val_labels}, batch_size=1)
        else:
            self.io.print_error('Say whaaaat ?! This model type is not supported : {}'.format(self.model_type))
        #if
        """

        self.eval(n_iter, val_images, val_labels, tag='VALIDATION')

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
