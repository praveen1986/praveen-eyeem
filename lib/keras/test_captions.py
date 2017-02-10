import os, sys
import yaml
import logging
import cStringIO

import json
import cv2
from PIL import Image
import time
import keras
import numpy as np
from io import BytesIO
import urlparse, urllib, StringIO
from scipy.misc import imresize
from sklearn.externals import joblib

from rnd_libs.lib.label_embedding.embedding_io import EmbeddingIO
from rnd_libs.lib.keras.lmdb_parser import LMDBParser
from rnd_libs.models.keras.models import model_dict
from keras import backend as K

logger = logging.getLogger(__name__)


class TheanoTesterCaptions():

    def __init__(self, config_file, verbose=False, raw_predictions=False):

        if not os.path.exists(config_file):
            print 'Error : could not find config file'.format(config_file)
            self.init = False
            return

        self.tester_config_file = config_file
        self.verbose = verbose
        self.raw_predictions = raw_predictions
        self.read_config()

        self.io = EmbeddingIO(self.cfgs['log_dir'], 'testing')

        self.tester_name = self.cfgs['model_name']
        self.model_config_file = self.cfgs['model_config_file']

        self.init = False
        self.thresholds = None

        mapping_keys = [d[0] for d in self.cfgs['mapping_list']]
        mapping_values = [d[1] for d in self.cfgs['mapping_list']]

        self.mapping = dict(zip(mapping_keys, mapping_values))

        if self.tester_name not in model_dict.keys():
            self.io.print_error('Not a valid model type {0} chose among {1}'.format(self.tester_name, model_dict.keys()))
            self.init = False
            return

        if not os.path.exists(self.tester_config_file):
            self.io.print_error('Could not find configuration file {0}'.format(self.tester_config_file))
            self.init = False
            return

        self.tester = None
        self.action_items = {'test': self.run, 'eval': self.eval}

    def _read_threshold_file(self):

        if 'threshold_file' in self.cfgs.keys():

            try:

                pfile = open(self.cfgs['threshold_file'], 'r')
                thresholds = json.load(pfile)
                pfile.close()

                self.thresholds = {c: thresholds[c] if c in thresholds.keys() else self.cfgs['global_threshold'] for c in self.concepts}

            except Exception as err:

                self.io.print_error('Error parsing threshold file {0},{1}'.format(self.cfgs['threshold_file'], err))

    def read_config(self):

        pfile = open(self.tester_config_file)
        self.cfgs = yaml.load(pfile)
        pfile.close()

    def setup(self):

        self.tester = model_dict[self.tester_name]()
        self.tester.configure(self.model_config_file)

        self.tester.define()

        self.model_type = self.tester.model.get_config()['name']

        self.im_h, self.im_w = self.tester.cfgs['image_height'], self.tester.cfgs['image_width']

        if not self.tester.init:
            self.io.print_error('Error with model definition')
            self.init = False
            return

        self.io.print_info('{0} Model defined from {1}'.format(self.tester_name, self.model_config_file))

        self.compile()

        if not self.tester.init:
            self.io.print_error('Error with model compilation')
            self.init = False
            return

        self.io.print_info('{0} Model compiled'.format(self.tester_name))

        self.init = True
        self.load_synset_file()
        self.load_mean_file()
        self._read_threshold_file()
        self.load_confidence_model()
        self.read_label_embedding()

    def compile(self):

        """
            This compile version is to be used for testing only. Since its optimized for predictions
        """

        self.output_nodes = self.cfgs['output_node_name']['test']

        if self.model_type == 'Sequential':
            self.inference = K.function([self.tester.model.get_input(train=False)], [self.tester.model.layers[-1].get_output(train=False)])
        elif self.model_type == 'Graph':
            self.inference = K.function([self.tester.model.get_input(train=False)], [self.tester.model.nodes[node].get_output(train=False) for node in self.output_nodes])
        else:
            self.io.print_error('Say whaaaat ?! This model type is not supported : {}'.format(self.model_type))
            sel.init = False

    def read_label_embedding(self):

        self.label_embedding = None
        self.label_vectors = None

        if 'label_embedding_file' in self.cfgs.keys():

            self.label_embedding = self.io.read_npz_file(self.cfgs['label_embedding_file'], 'matrix')
            self.label_vectors = self.io.read_npz_file(self.cfgs['label_embedding_file'], 'vectors')

            self.io.print_info('Read label embedding file {0}'.format(self.cfgs['label_embedding_file']))

    def load_confidence_model(self):

        self.confidence_model = None

        if 'confidence_model' in self.cfgs.keys():
            self.confidence_model = joblib.load(self.cfgs['confidence_model'])

    def run(self):

        self.load_synset_file()

        if not self.init:
            return

        self.test_images = self.io.read_images(self.cfgs['test_file_list'])
        image_batches = [self.test_images[i:i + self.cfgs['batch_size']] for i in range(0, len(self.test_images), self.cfgs['batch_size'])]

        self.io.print_info('{0} images split into {1} Batches of {2}'.format(len(self.test_images), len(image_batches), self.cfgs['batch_size']))

        for idx, images in enumerate(image_batches):

            if not self.cfgs['force']:

                flag = self.check_jsons(images)

                if flag:
                    self.io.print_warning('Skipping batch {0} of {1}'.format(idx, len(image_batches)))
                    continue

            pixels, basenames = self.fetch_images(images)
            flag, predictions, confidence, t = self.classify_images(pixels)

            if self.raw_predictions is False:
                getty_like_filter = [self.suppress_stop_list(g) for g in predictions]
                getty_like_safe = [self.run_thresholds(g) for g in getty_like_filter]
                getty_like_pc = [self.map_concepts(g) for g in getty_like_safe]
                getty_like_public = [self.resolve_antonyms(g) for g in getty_like_pc]
                getty_like_unique = [self.resolve_duplicates(g) for g in getty_like_public]
                predictions_pc = getty_like_unique
                save_ext = '.json'
            else:
                predictions_pc = predictions
                save_ext = '.npz'

            if not flag:
                continue

            file_paths = [os.path.join(self.cfgs['results_dir'], im + save_ext) for im in basenames]

            if not self.raw_predictions:
                to_save = [dict({'external': p}) for p in predictions_pc]
            else:
                to_save = predictions_pc

            assert(len(file_paths) == len(to_save))

            self.io.print_info('Done batch {0} of {1} in {2} secs'.format(idx, len(image_batches), t))

            f = map(self.io.save_good_labels, file_paths, to_save)

    def check_jsons(self, images):

        basenames = []

        for image in images:

            im_basename = os.path.basename(image)
            im_basename, _ = os.path.splitext(im_basename)

            basenames.append(im_basename)

        flags = [os.path.exists(os.path.join(self.cfgs['results_dir'], b + '.json')) for b in basenames]

        return np.all(flags)

    #
    # Which measure to keep
    # Precision/Recall/F1 scores
    #

    def eval(self):

        self.load_synset_file()

        self.test_images = self.io.read_images(self.cfgs['val_file_list'])

        self.predictions = {}
        self.gt = {}

        for idx, image in enumerate(self.test_images):

            if idx % 1000 == 0:
                self.io.print_info('Reading {0}/{1}'.format(idx, len(self.test_images)))

            im_basename = os.path.basename(image)
            im_basename, _ = os.path.splitext(im_basename)

            result_json = os.path.join(self.cfgs['results_dir'], im_basename + '.json')
            gt_json = os.path.join(self.cfgs['gt_dir'], im_basename + '.json')

            p, s = self.io.read_prediction_json(result_json)
            g = self.io.read_vector(gt_json)

            if p is [] or g is []:
                continue

            self.predictions[im_basename] = zip(p, s)
            self.gt[im_basename] = list(set(g).intersection(self.concepts))

        for t in self.cfgs['thresholds']:
            self.compute_precision_recall(t)

    def compute_precision_recall(self, thresh):

        fps = {}
        tps = {}
        fns = {}
        gt_dict = {}

        for c in self.concepts:
            fps[c], tps[c], fns[c], gt_dict[c] = [], [], [], []

        for key, value in self.predictions.items():

            detections = [v[0] for v in value if v[1] >= thresh]
            gt = self.gt[key]

            for g in gt:
                gt_dict[g].append(key)

            tp = list(set(gt).intersection(detections))
            fp = list(set(detections) - set(gt))
            fn = list(set(gt) - set(detections))

            for t in tp:
                tps[t].append(key)

            for f in fp:
                fps[f].append(key)

            for n in fn:
                fns[n].append(key)

        for c in self.concepts:

            if gt_dict[c] == []:
                self.io.print_info('Skipping {0}, no samples in ground-truth'.format(c))
                continue

            p = len(tps[c]) / (len(tps[c]) + len(fps[c]) + 0.00000001)
            r = len(tps[c]) / (len(gt_dict[c]) + 0.0)

            f1 = self.compute_f1(p, r)

            self.io.print_info('At {0} : {1} precision {2}, recall {3}, f1-score {4}'.format(c, thresh, p, r, f1))

    def compute_f1(self, p, r):

        return 2 * (p * r) / (p + r + 0.000001)

    def load_synset_file(self):

        pfile = open(self.cfgs['synset_file'], 'r')
        concepts = pfile.readlines()
        pfile.close()

        self.concepts = [p.strip() for p in concepts]

    def load_mean_file(self):

        if not self.cfgs['mean_file'] == 'None':
            pfile = open(self.cfgs['mean_file'], 'r')
            m = np.load(pfile)
            pfile.close()
        else:
            m = [[[128.0]], [[128.0]], [[128.0]]]

        m = np.mean(np.mean(m, axis=1), axis=1)

        self.mean_pixel_value = np.array([m[0], m[1], m[2]])

        self.pixel_scaling = self.cfgs['pixel_scaling']['test'] if 'pixel_scaling' in self.cfgs.keys() else None

        self.channel_swap = self.cfgs['channel_swap']['test'] if 'channel_swap' in self.cfgs.keys() else None

        self.image_shuffle = self.cfgs['image_shuffle']['test'] if 'image_shuffle' in self.cfgs.keys() else None

    def jsonify(self, results):

        return [[r[0], str(r[1])] for r in results]

    def map_to_synsets(self, results):

        results = {r[0]: r[1] for r in results}

        return [results[c] for c in self.concepts]

    def parse_predictions(self, p):

        sort_idx = np.argsort(-p)

        concepts = [(self.concepts[idx], float(p[idx])) for idx in sort_idx[:self.cfgs['top_k']['test']]]

        return concepts

    def classify_images(self, images):

        confidence = [None]

        if images == []:
            return False, [(), ()], 0.0

        try:

            tstart = time.time()

            (predictions, features) = self.inference(images)

            if self.label_vectors is not None:
                predictions /= np.linalg.norm(predictions, axis=1)[:, np.newaxis]
                predictions = np.dot(predictions, self.label_vectors)

            if self.confidence_model is not None:
                confidence = [('Confidence', p) for p in self.confidence_model.predict(features)]

            tend = time.time()

        except Exception as err:
            self.io.print_error('Error processing image, {0}'.format(err))
            return False, [(), ()], None, 0.0

        if not self.raw_predictions:
            results = True, [self.parse_predictions(p) for p in predictions], confidence, '{0:.4f}'.format((tend - tstart))
        else:
            results = True, predictions, confidence, '{0:.4f}'.format((tend - tstart))

        return results

    def predict(self, image):

        image = self.normalize_images([image])

        return self.classify_images(image)

    def normalize_images(self, images):

        normalize_images = []

        for image in images:

            image *= 255.0

            if self.cfgs['square_crop']:
                if self.verbose:
                    self.io.print_info('Yo ! As you like it - Squaring up the image')
                image = self.square_it_up(image)

            try:
                image = cv2.resize(image, (self.im_h, self.im_w))
            except Exception as err:
                self.io.print_error('Could not parse test image {0}'.format(e))
                return False, [], 0.0

            image -= self.mean_pixel_value
            image = image.transpose(2, 0, 1)

            # Channel swap since caffe.io.load_image returns RGB image and training was done with BGR
            image = image[(2, 1, 0), :, :]

            normalize_images.append(image)

        return [np.array(normalize_images)]

    def scale_image(self, image):

            #
            # caffe scales image 0-1 in pixel values we bring it back to 255 as keras likes it that way
            #

            # No more caffe no more scaling down to 0-1 in caffe
            # image *= 255.0

            tstart = time.time()

            if self.cfgs['square_crop']:
                if self.verbose:
                    self.io.print_info('Yo ! As you like it - Squaring up the image')
                image = self.square_it_up(image)

            tend = time.time()

            tstart = time.time()

            try:
                image = cv2.resize(image, (self.im_h, self.im_w)).astype(np.float32)
            except Exception as err:
                self.io.print_error('Could not parse {0}'.format(e))
                return None

            tend = time.time()

            tstart = time.time()

            if self.mean_pixel_value is not None:
                image -= self.mean_pixel_value

            if self.pixel_scaling is not None:
                image /= self.pixel_scaling

            if self.channel_swap is not None:
                image = image[:, :, self.channel_swap]

            # H,W,C to C,H,W
            if self.image_shuffle is not None:
                image = image.transpose(self.image_shuffle)

            tend = time.time()

            return image

    def fetch_images(self, image_file_names):

        images = []
        basenames = []

        tstart = time.time()

        for idx, image_file_name in enumerate(image_file_names):

            im_basename = os.path.basename(image_file_name)
            im_basename, _ = os.path.splitext(im_basename)

            if not urlparse.urlparse(image_file_name).scheme == "":

                url_response = urllib.urlopen(image_file_name)

                if url_response.code == 404:
                    print self.io.print_error('[Testing] URL error code : {1} for {0}'.format(image_file_name, url_response.code))
                    continue

                try:

                    string_buffer = StringIO.StringIO(url_response.read())
                    image = np.asarray(bytearray(string_buffer.read()), dtype="uint8")

                    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                    image = image[:, :, (2, 1, 0)]
                    image = image.astype(np.float32)

                except Exception as err:

                    print self.io.print_error('[Testing] Error with URL {0} {1}'.format(image_file_name, err))
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

                    print self.io.print_error('[Testing] Error with image file {0} {1}'.format(image_file_name, err))
                    continue
            else:

                try:

                    image = self.conn.read_image(image_file_name)

                    if image is None:
                        raise self.io.print_error("[Testing] Image data could not be loaded - %s" % image_file_name)
                        continue

                    image = np.array(image)

                    if image.ndim == 2:
                        image = image[:, :, np.newaxis]
                        image = np.tile(image, (1, 1, 3))
                    elif image.shape[2] == 4:
                        image = image[:, :, :3]

                except Exception as err:

                    print self.io.print_error('[Testing] Error with S3 Bucket hash {0} {1}'.format(image_file_name, err))
                    continue

            # try : do or do not there is not try

            image = self.scale_image(image)

            if image is None:
                continue

            if self.verbose:
                self.io.print_info('Processing {0} {1}'.format(image_file_name, image.shape))

            images.append(image)

            basenames.append(im_basename)

        tend = time.time()

        self.io.print_info('Fetching {0} images took {1:.4f} secs'.format(len(image_file_names), tend - tstart))

        return [np.array(images)], basenames

    def fetch_gif_frames(self, inGif):

        # Only works for URLs for now
        # write frames to disk and pass filenames to fetch_images

        outFolder = '/tmp'

        frame = Image.open(BytesIO(urllib.urlopen(inGif).read()))
        nframes = 0

        file_names = []

        while frame:
            frame_name = '{0}/{1}-{2}.png'.format(outFolder, os.path.basename(inGif), nframes)
            frame.save(frame_name, 'png')
            file_names.append(frame_name)
            nframes += 1
            try:
                frame.seek(nframes)
            except EOFError:
                break

        file_names = [file_names[i] for i in range(0, len(file_names), 10)]

        return file_names

    def non_max(self, predictions):

        concepts = {}

        for pred in predictions:
            for p in pred:
                if p[0] in concepts.keys():
                    concepts[p[0]].append(float(p[1]))
                else:
                    concepts[p[0]] = [float(p[1])]

        for key, value in concepts.items():
            concepts[key] = np.max(np.array(value))

        labels = concepts.keys()
        scores = [concepts[l] for l in labels]

        sort_idx = np.argsort(-np.array(scores))

        results = [(labels[idx], str(scores[idx])) for idx in sort_idx]

        return [results]

    def square_it_up(self, image):

        edge_size = np.min(image.shape[:2])
        cy, cx = np.array(image.shape[:2]) // 2

        r_img = image[np.max([0, cy - edge_size // 2]):np.min([cy + edge_size // 2, image.shape[0]]), np.max([0, cx - edge_size // 2]):np.min([cx + edge_size // 2, image.shape[1]]), :]

        return r_img

    def resolve_duplicates(self, concepts):

        seen = set()
        unique = []

        for c in concepts:
            if c[0] in seen:
                logger.debug("Suppressing duplicate {}:{}".format(c[0], c[1]))
                continue
            seen.add(c[0])
            unique.append(c)

        return unique

    def filter_concepts(self, predictions):

        return [p[:self.cfgs['store_top_k']] for p in predictions]

    def suppress_stop_list(self, predictions):

        return [c for c in predictions if c[0] not in self.cfgs['stop_list_']]

    def run_thresholds(self, predictions):

        return self.threshold_concepts(predictions)

    def resolve_antonyms(self, human_predictions):

        conflicting_predictions = []
        human_dict = {d[0]: d[1] for d in human_predictions}

        human_labels = [d[0] for d in human_predictions]
        human_scores = [d[1] for d in human_predictions]

        for c1, c2 in self.cfgs['antonym_list']:

            if not (c1 in human_labels and c2 in human_labels):
                continue
            # if

            s1, s2 = human_dict[c1], human_dict[c2]
            idx = human_labels.index(c2) if s1 > s2 else human_labels.index(c1)

            logger.debug('Suppressing {0}:{1}'.format(human_labels[idx], human_scores[idx]))

            del human_labels[idx]
            del human_scores[idx]

        # for

        remove_flag = -1

        for idx, group in enumerate(self.cfgs['count_order_list']):

            _this = np.intersect1d(human_dict.keys(), group)

            if len(_this) > 0:
                remove_flag = idx + 1
                break
                # if

        # for

        if not remove_flag == len(self.cfgs['count_order_list']):

            remove_tags = []

            for g in self.cfgs['count_order_list'][remove_flag:]:
                remove_tags.extend(g)
            # for

            for t in remove_tags:

                if t not in human_labels:
                    continue
                # if

                ridx = human_labels.index(t)

                del human_labels[ridx]
                del human_scores[ridx]

                # for

        # if

        return [(g, s) for g, s in zip(human_labels, human_scores)]

    # def

    def map_concepts(self, concepts):

        mapped_predictions = [(self.mapping[c[0]], c[1]) if c[0] in self.mapping.keys() else c for c in concepts]

        mapped_predictions = self.conditional_mapping(mapped_predictions)

        return mapped_predictions

    # def

    def conditional_mapping(self, predictions):

        if 'conditional_list' in self.cfgs.keys():

            for condition, mapping in self.cfgs['conditional_list']:

                concepts = [p[0] for p in predictions]

                holds = list(set(concepts).intersection(set(condition)))

                if not len(holds) == 0:
                    predictions = [p for p in predictions if not p[0] in mapping]
                    # if

                    # for

        # if

        return predictions

    # def

    def threshold_concepts(self, predictions):

        if self.thresholds is not None:
            predictions = [p for p in predictions if p[1] >= self.thresholds[p[0]]]

        # if

        return predictions

        # def
