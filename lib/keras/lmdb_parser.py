# LMDB file parser
import pickle
import os, sys
import lmdb
import numpy as np
import cv2
from scipy.misc import imresize
from rnd_libs.lib.keras.helpers import pad_sequences
from rnd_libs.lib.keras.lmdb_pb2 import Datum
from rnd_libs.lib.label_embedding.embedding_io import EmbeddingIO

#
# These are not actually LMDB files but HDF5 files :D
#

#
# None - Original,
# drop_out - drop labels from a batch with inverse frequency of the label
# mikolov - random class sample, random instance sample, 1 label per data point
# mikolov_drop_out - self evident
# mikolov_original - random class sample, random instance sample, keep all labels
#


class LMDBParser():

    def __init__(self, lmdb_dir_path, log_dir='/tmp', keep=32):

        # 1. Path to directory containing data.mdb
        # 2. keep which will take 1000 records from lmdb file into CPU memory

        self.lmdb_dir_path = lmdb_dir_path
        self.keep = keep
        self.passes = 0
        self.datum_seen = 0
        self.mean_weight = 1.4
        self.label_distribution = None
        self.label_statistics = None
        self.label_statistics_train = None

        self.io = EmbeddingIO(log_dir, log_type='lmdb-parsing-{0}'.format(lmdb_dir_path.split('/')[-1]))

        if not os.path.exists(self.lmdb_dir_path):
            self.io.print_error('Could not find LMDB path {0}'.format(self.lmdb_dir_path))
            self.init = False
            return

        try:

            self.lmdb_env = lmdb.open(self.lmdb_dir_path, readonly=True, lock=False)
            self.lmdb_txn = self.lmdb_env.begin()

            self.lmdb_cursor = self.lmdb_txn.cursor()
            self.read_label = self.lmdb_txn.get
            self.lmdb_cursor.first()

            self.datum = Datum()
            self.lmdb_entries = int(self.lmdb_txn.stat()['entries'])

        except Exception as e:

            self.io.print_error('Error reading LDMB file {0},{1}'.format(self.lmdb_dir_path, e))
            self.init = False
            return

        self.io.print_info('Train LMDB file : {0} '.format(self.lmdb_dir_path))
        self.init = True

        self.get_batch = {
            'None': self.get_next_slice,
            'drop_out': self.get_next_slice,
            'mikolov': self.get_next_slice_sampled,
            'mikolov_original': self.get_next_slice_sampled,
            'mikolov_drop_out': self.get_next_slice_sampled,
            'mikolov_em_drop_out': self.get_next_slice_sampled,
            'mikolov_tetris_drop_out': self.get_next_slice_sampled,
            'smote': self.get_next_slice_smote,
            'captions': self.get_next_slice_captions,
            'captions_mikolov': self.get_next_slice_captions_mikolov}

    def reset_data_stats(self):

        self.io.print_info('Mean weight for sampling {}'.format(self.mean_weight))

        self.label_distribution = np.ones((len(self.vocabulary),)) / (len(self.vocabulary) + 0.0)
        self.label_statistics = np.zeros((len(self.vocabulary),))
        self.label_statistics_train = np.zeros((len(self.vocabulary),))
    def set_params_test(self,vocabulary):
        self.vocabulary=vocabulary
    def set_params(
            self, crop_size, mean_pixel_value, keep, channel_swap,
            pixel_scaling, label_sampling, vocabulary, label_mapping,
            max_overlap=4, image_shuffle=[2, 0, 1], label_reduce=None,
            label_embedding=None):

        self.keep = keep
        self.crop_size = crop_size
        self.mean_pixel_value = mean_pixel_value
        self.vocabulary = vocabulary
        self.label_mapping = label_mapping
        self.channel_swap = channel_swap
        self.pixel_scaling = pixel_scaling
        self.label_sampling = label_sampling
        self.image_shuffle = image_shuffle
        self.label_reduce = label_reduce
        self.max_overlap = max_overlap
        self.label_embedding = label_embedding
        self.reset_data_stats()
        self.build_dataset_cdf()

    def set_captions_params(self, LUT=None, max_caption_len=None,
                            caption_vocabulary=None, label_sampling=None):

            self.LUT = LUT
            self.max_caption_len = max_caption_len
            self.caption_vocabulary = caption_vocabulary
            self.label_sampling = label_sampling
            self.START = self.LUT['BOS']
            self.END = self.LUT['EOS']
            self.PAD = self.LUT['PAD']

    def build_dataset_cdf(self):

        if self.label_mapping is None:
            return

        dist = [(k, len(v)) for k, v in self.label_mapping.items() if k in self.vocabulary]
        concepts = self.label_mapping['vocabulary']
        concepts_count = [len(d) for d in self.label_mapping['mapping']]
        sort_idx = np.argsort(-np.array(concepts_count))

        sorted_concepts = [concepts[idx] for idx in sort_idx]

        concepts_pdf = np.array(concepts_count) / (np.sum(concepts_count) + 0.0)
        concepts_cdf = np.cumsum(concepts_pdf)

        self.concepts_pdf = np.array([concepts_pdf[sorted_concepts.index(c)] for c in self.vocabulary])
        self.concepts_cdf = np.array([concepts_cdf[sorted_concepts.index(c)] for c in self.vocabulary])

        self.concepts_cdf_norm = self.concepts_cdf / np.sum(self.concepts_cdf)

        self.io.print_info('Training set statistics set up')

    # def build_dataset_cdf(self):

    #     if self.label_mapping is None:
    #         return

    #     dist = [(k, len(v)) for k, v in self.label_mapping.items() if k in self.vocabulary]
    #     concepts = [d[0] for d in dist]
    #     concepts_count = [d[1] for d in dist]
    #     sort_idx = np.argsort(-np.array(concepts_count))

    #     sorted_concepts = [concepts[idx] for idx in sort_idx]

    #     concepts_pdf = np.array(concepts_count) / (np.sum(concepts_count) + 0.0)
    #     concepts_cdf = np.cumsum(concepts_pdf)

    #     self.concepts_pdf = np.array([concepts_pdf[sorted_concepts.index(c)] for c in self.vocabulary])
    #     self.concepts_cdf = np.array([concepts_cdf[sorted_concepts.index(c)] for c in self.vocabulary])

    #     self.concepts_cdf_norm = self.concepts_cdf / np.sum(self.concepts_cdf)

    #     self.io.print_info('Training set statistics set up')

    def get_label_Frimgfile(self,filename):
        
        dirname='/nas/datasets/getty-collection/getty-all/'
        key=dirname+filename
        value = self.lmdb_txn.get(key)
        #import ipdb; ipdb.set_trace()
        if not (value==None): 
            self.datum.ParseFromString(value)
            lab = self.datum.label
        else:
            lab=[]
        return lab
    def get_labels(self,f,ID):
        #import ipdb; ipdb.set_trace()
        stream = f(ID)
        return pickle.loads(stream)

    def create_lmdb_keys_labels(self):
        import ipdb; ipdb.set_trace()
        
        l=0
        env = lmdb.open("./keywording_test_images_labels", map_size=1e12)
        try:
            with env.begin(write=True) as txn:
                while 1:
                    keys,labels=self.get_next_slice_img_lab()
                    pickled_labels = [pickle.dumps(label, protocol=1) for label in labels]
                    import ipdb; ipdb.set_trace()
                    if self.passes<1:
                        for i,key in enumerate(keys):
                            ID,_=os.path.splitext(os.path.basename(key))
                            if pickled_labels[i]  != None:   
                                txn.put(ID.encode('ascii'), pickled_labels[i])
                                l=l+1
                                print "Processes %d" % l
                        #break
                    else:
                        break
        except Exception as e:
            print 'problem processing image %s' % ID
        import ipdb; ipdb.set_trace()
    def create_keys_labels_dict(self):
        import ipdb; ipdb.set_trace()
        keys_labels=dict()
        l=0
        
        import ipdb; ipdb.set_trace()

            
        while 1:
            keys,labels=self.get_next_slice_img_lab()
                    #pickled_labels = [pickle.dumps(label, protocol=1) for label in labels]
                    
            if self.passes<1:
                for i,key in enumerate(keys):
                    ID,_=os.path.splitext(os.path.basename(key))
                    if len(labels[i])  == len(self.vocabulary):   
                        keys_labels[ID]=labels[i]
                        print keys_labels[ID].sum()
                        l=l+1
                        #print "Processes %d" % l
                        #break
                    else:
                        print "Not Processes image %s" % ID
                        break
            else:
                break
        import ipdb; ipdb.set_trace()
        pickle.dump(keys_labels, open('/nas2/praveen/home/rnd-libs/rnd_libs/lib/keras/key_labels', 'wb'))
        import ipdb; ipdb.set_trace()
    def get_next_slice_img_lab(self):

        # Warning : This is hard coded for inception_v3 model should be set from config file
        labels = np.zeros((self.keep, len(self.vocabulary)),dtype='u1')
        keys = []

        # TODO : HSS - setting fields from YAML
        # [HSS] get rid of while loops with cleaner sampling file

        i = 0

        while i < self.keep:

            # if End of LMDB file start from the begining

            if not self.lmdb_cursor.next():
                self.lmdb_cursor.first()
                self.passes += 1
                self.io.print_info('Start of next pass over the test set {0}'.format(self.passes))

            # Read a key-value record from the queue

            try:

                key, value = self.lmdb_cursor.item()
                self.datum.ParseFromString(value)
                lab = self.datum.label

                if lab is None:
                    self.io.print_warning('{1}/{2} Skipping {0}, no useful label found'.format(key, i, self.keep))
                    continue

        
                labels[i, :] = lab
                keys.append(key)

                i += 1

            except Exception as e:

                self.io.print_info('Skipping corrupted images, {0}'.format(e))

        # update label statistics and sampling distribution
        #self.update_label_stats(labels)

        test_labels_full = labels
        #test_labels_semi = self.label_drop(keys, test_labels_full) if 'drop_out' in self.label_sampling else test_labels_full

        # if self.label_embedding is not None:
        #     train_labels_full = np.dot(test_labels_full, self.label_embedding)
        #     train_labels_semi = np.dot(test_labels_semi, self.label_embedding)
        # else:
        #     train_labels_full = test_labels_full
        #     train_labels_semi = test_labels_semi

        return keys, test_labels_full
    def get_next_slice(self):

        # Warning : This is hard coded for inception_v3 model should be set from config file
        images = np.zeros((self.keep, 3, self.crop_size[0], self.crop_size[1]))
        labels = np.zeros((self.keep, len(self.vocabulary)))
        keys = []

        # TODO : HSS - setting fields from YAML
        # [HSS] get rid of while loops with cleaner sampling file

        i = 0

        while i < self.keep:

            # if End of LMDB file start from the begining

            if not self.lmdb_cursor.next():
                self.lmdb_cursor.first()
                self.passes += 1
                self.io.print_info('Start of next pass over the training set {0}'.format(self.passes))

            # Read a key-value record from the queue

            try:

                key, value = self.lmdb_cursor.item()

                arr, lab = self.get_image_label(value)

                if lab is None:
                    self.io.print_warning('{1}/{2} Skipping {0}, no useful label found'.format(key, i, self.keep))
                    continue

                images[i, :, :, :] = arr
                labels[i, :] = lab
                keys.append(key)

                i += 1

            except Exception as e:

                self.io.print_info('Skipping corrupted images, {0}'.format(e))

        # update label statistics and sampling distribution
        self.update_label_stats(labels)

        test_labels_full = labels
        test_labels_semi = self.label_drop(keys, test_labels_full) if 'drop_out' in self.label_sampling else test_labels_full

        if self.label_embedding is not None:
            train_labels_full = np.dot(test_labels_full, self.label_embedding)
            train_labels_semi = np.dot(test_labels_semi, self.label_embedding)
        else:
            train_labels_full = test_labels_full
            train_labels_semi = test_labels_semi

        return keys, images, train_labels_semi, test_labels_full

    def get_next_slice_sampled(self):

        # Warning : This is hard coded for inception_v3 model should be set from config file
        images = np.zeros((self.keep, 3, self.crop_size[0], self.crop_size[1]))
        labels = np.zeros((self.keep, len(self.vocabulary)))
        pseudo_labels = np.zeros((self.keep, len(self.vocabulary)))
        keys = []

        #
        # while since some smaples might not be in this lmdb
        #

        i = 0

        #
        # [HSS] get rid of while loops with cleaner sampling file
        #

        while i < self.keep:

            # sample random class

            if 'tetris' in self.label_sampling:

                # sample classes which havent been seen
                if self.datum_seen == 0:
                    random_class = np.random.choice(self.vocabulary, 1)[0]
                else:
                    prob = 1.0 - self.label_statistics_train / (np.max(self.label_statistics_train) + 0.0)
                    prob /= np.sum(prob)
                    random_class = np.random.choice(self.vocabulary, 1, p=prob)[0]

            else:
                random_class = np.random.choice(self.label_mapping['vocabulary'], 1)[0]

            # get the class index to build the vector later
            random_class_index = self.label_mapping['vocabulary_mapping'][random_class]

            # select random image sample for the sampled class
            image_index = np.random.choice(self.label_mapping['mapping'][random_class_index], 1)[0]

            # fetching actual image string from integer to image mapping
            key = str(self.label_mapping['images'][image_index])
            # casting to str is important, because sometimes the key is unicode

            # look up in LMDB for image-data
            value = self.lmdb_txn.get(key)

            # skip if not in current lmdb
            if value is None:
                continue

            self.datum_seen += 1

            if self.datum_seen >= self.lmdb_entries:
                self.datum_seen = 0
                self.passes += 1
                self.io.print_info('Start of next pass over the training set {0}'.format(self.passes))
                self.reset_data_stats()

            try:

                arr, lab = self.get_image_label(value)

                if lab is None:
                    self.io.print_warning('{1}/{2} Skipping {0}, no useful label found'.format(key, i, self.keep))
                    continue

                nlab = np.zeros_like(lab)

                #
                # set random class index to 1 rest to 0
                #

                nlab[random_class_index] = 1.0

                if len(lab) == 0:
                    continue

                images[i, :, :, :] = arr
                labels[i, :] = lab
                pseudo_labels[i, :] = nlab
                keys.append(key)

                i += 1

            except Exception as e:

                self.io.print_info('Skipping corrupted images, {0}'.format(e))

        # update label statistics and sampling distribution
        self.update_label_stats(labels)

        base_labels = pseudo_labels if self.label_sampling == 'mikolov' else labels
        base_labels = self.label_drop(keys, base_labels) if 'drop_out' in self.label_sampling else base_labels

        test_labels_full = labels

        self.label_statistics_train += np.sum(base_labels, axis=0)

        if self.label_embedding is not None:
            train_labels_semi = np.dot(base_labels, self.label_embedding)
        else:
            train_labels_semi = base_labels

        return keys, images, train_labels_semi, test_labels_full

    def update_label_stats(self, labels):

        self.label_statistics += np.sum(labels, axis=0)
        self.label_distribution[np.where(self.label_statistics)] /= self.label_statistics[np.where(self.label_statistics)]
        self.label_distribution /= np.sum(self.label_distribution)

    def get_next_slice_smote(self):

        # Warning : This is hard coded for inception_v3 model should be set from config file
        images = np.zeros((self.keep, 3, self.crop_size[0], self.crop_size[1]))
        labels = np.zeros((self.keep, len(self.vocabulary)))
        all_labels = np.zeros((self.keep, len(self.vocabulary)))
        keys = []

        #
        # while since some smaples might not be in this lmdb
        #

        i = 0

        #
        # [HSS] get rid of while loops with cleaner sampling file
        #

        current_labels = None

        while i < self.keep:

            # sample random class
            random_class = np.random.choice(self.vocabulary, 1)[0]
            # self.io.print_info('Sampled {0}'.format(random_class))

            # get the class index to build the vector later
            random_class_index = self.vocabulary.index(random_class)

            # select random image sample for the sampled class
            key = str(np.random.choice(self.label_mapping[random_class], 1)[0])
            # casting to str is important, because sometimes the key is unicode

            # look up in LMDB for image-data
            value = self.lmdb_txn.get(key)

            # skip if not in current lmdb
            if value is None:
                continue

            self.datum_seen += 1

            if self.datum_seen >= self.lmdb_entries:
                self.datum_seen = 0
                self.passes += 1
                self.io.print_info('Start of next pass over the training set {0}'.format(self.passes))

            try:

                arr, lab = self.get_image_label(value)

                if lab is None:
                    self.io.print_warning('{1}/{2} Skipping {0}, no useful label found'.format(key, i, self.keep))
                    continue

                # current_labels, flag = self.is_good_to_add(current_labels,lab)
                current_labels, lab, true_labels, flag = self.get_good_labels(current_labels, lab)

                if not flag:
                    continue

                images[i, :, :, :] = arr
                labels[i, :] = lab
                all_labels[i, :] = true_labels
                keys.append(key)

                i += 1

                # self.io.print_info('Current batch size {}'.format(len(images)))

            except Exception as e:

                self.io.print_info('Skipping corrupted images, {0}'.format(e))

        return keys, images, labels, all_labels

    def is_good_to_add(self, current, lab):

        if current is None:
            return np.copy(lab), True

        dot = np.dot(current, lab)
        # self.io.print_info('Dot product {0}'.format(dot))

        if dot >= self.max_overlap:
            return np.copy(current), False
        else:
            current = np.bitwise_or(current, lab)
            # self.io.print_info('Used labels {0}'.format(np.sum(current)))
            return np.copy(current), True

    def get_good_labels(self, current, lab):

        all_labels = np.copy(lab)

        if current is None:
            return np.copy(lab), np.copy(lab), all_labels, True

        new_lab = np.bitwise_and(lab, np.invert(np.bitwise_and(current, lab)))

        if np.sum(new_lab) == 0:
            return np.copy(current), np.copy(lab), all_labels, False
        else:
            current = np.bitwise_or(current, lab)
            return np.copy(current), np.copy(new_lab), all_labels, True

    def get_image_label(self, value):

        self.datum.ParseFromString(value)

        arr = np.fromstring(self.datum.data, dtype=np.uint8).reshape(self.datum.channels, self.datum.height, self.datum.width)
        arr = arr.transpose(1, 2, 0)

        # This casting is required since uint8 and float32 cause an under-flow while mean subtraction below
        arr = np.asarray(arr, dtype=np.float32)

        if self.crop_size is not None:
            arr = cv2.resize(arr, self.crop_size).astype(np.float32)

        if self.mean_pixel_value is not None:
            arr -= self.mean_pixel_value

        if self.channel_swap is not None:
            arr = arr[:, :, self.channel_swap]

        if self.pixel_scaling is not None:
            arr /= self.pixel_scaling

        # As theano likes it N_channels x image_H x image_w

        if self.image_shuffle is not None:
            arr = arr.transpose(self.image_shuffle)

        # Label is a single field but could be an array
        lab = self.datum.label

        if self.label_reduce is not None:
            dlab = np.dot(self.label_reduce, lab)
            lab = np.zeros_like(dlab)
            lab[np.where(dlab)[0]] = 1
            if np.sum(lab) == 0:
                lab = None

        return arr, lab

    def label_drop(self, keys, labels):

        if 'em' in self.label_sampling:

            # sample inversly to what network has seen in the entire history
            freq = self.label_statistics_train + 0.0
            non_empty_inices = np.where(self.label_statistics_train)

            if not non_empty_inices == []:
                freq[np.where(self.label_statistics_train > self.mean_weight * np.mean(self.label_statistics_train[non_empty_inices]))] = self.datum_seen
                freq /= self.datum_seen

        elif 'tetris' in self.label_sampling:

            residue = np.max(self.label_statistics_train) - self.label_statistics_train

            # reset if everything has been seen
            if np.sum(residue) == 0:
                return labels

            drop_labels = np.zeros_like(labels)

            # keep at most if available residue amount of samples for the label
            for idx, f in enumerate(residue):
                for idy in np.random.choice(np.where(labels[:, idx]), f):
                    drop_labels[idy][idx] = 1.0

            return np.multiply(labels, drop_labels)

        else:

            # sample inversly to batch frequency
            freq = np.sum(labels, axis=0) / (len(keys) + 0.0)
            # freq = 1 - self.concepts_cdf

        freq = [max(0, min(1, f)) for f in freq]

        drop_labels = np.array([np.random.binomial(1, 1 - f, len(keys)) for f in freq]).transpose()

        return np.multiply(labels, drop_labels)

    def get_next_slice_captions(self):

        images = np.zeros((self.keep, 3, self.crop_size[0], self.crop_size[1]))
        partial_captions = np.zeros((self.keep, self.max_caption_len))
        next_words = np.zeros((self.keep, self.max_caption_len + 1, len(self.caption_vocabulary)))
        keys = []

        caption_range = range(self.max_caption_len + 1)

        for i in range(self.keep):

            # if End of LMDB file start from the begining

            if not self.lmdb_cursor.next():
                self.lmdb_cursor.first()
                self.passes += 1
                self.io.print_info('Start of next pass over the training set {0}'.format(self.passes))

            # Read a key-value record from the queue

            try:

                key, value = self.lmdb_cursor.item()

                arr, cap = self.get_image_label(value)
                caption = [self.START] + list(cap) + [self.END]

                caption = pad_sequences([caption], maxlen=self.max_caption_len, padding='post', value=self.PAD)[0]

                images[i, :, :, :] = arr
                partial_captions[i, :] = caption
                next_words[i, caption_range, np.hstack((caption, [self.PAD]))] = 1
                keys.append(key)

            except Exception as e:

                self.io.print_info('Skipping corrupted images, {0}'.format(e))

        return keys, images, partial_captions, next_words

    def get_next_slice_captions_mikolov(self):

        images = np.zeros((self.keep, 3, self.crop_size[0], self.crop_size[1]))
        partial_captions = np.zeros((self.keep, self.max_caption_len))
        next_words = np.zeros((self.keep, self.max_caption_len + 1, len(self.caption_vocabulary)))
        keys = []

        i = 0

        caption_range = range(self.max_caption_len + 1)

        while i < self.keep:

            # if End of LMDB file start from the begining

            if not self.lmdb_cursor.next():
                self.lmdb_cursor.first()
                self.passes += 1
                self.io.print_info('Start of next pass over the training set {0}'.format(self.passes))

            # Read a key-value record from the queue

            try:

                random_class = np.random.choice(self.vocabulary, 1)[0]

                # get the class index to build the vector later
                random_class_index = self.vocabulary.index(random_class)

                # select random image sample for the sampled class
                key = str(np.random.choice(self.label_mapping[random_class], 1)[0])
                # casting to str is important, because sometimes the key is unicode

                # look up in LMDB for image-data
                value = self.lmdb_txn.get(key)
                arr, cap = self.get_image_label(value)

                if value is None:
                    continue

                if len(cap) + 2 > self.max_caption_len:
                    continue

                caption = [self.START] + list(cap) + [self.END]
                caption = pad_sequences([caption], maxlen=self.max_caption_len, padding='post', value=self.PAD)

                images[i, :, :, :] = arr
                partial_captions[i, :] = caption
                next_words[i, caption_range, np.hstack((caption, [self.PAD]))] = 1
                keys.append(key)

                i += 1

            except Exception as e:

                self.io.print_info('Skipping corrupted invalid images, {0}'.format(e))

        return keys, images, partial_captions, next_words

    def get_triplets(self, queries, crop_size=None, mean_value=None, keep=None):

        pass

    def check_passes(self):

        self.io.print_info('Done {0} passes of lmdb'.format(self.passes))

        return self.passes

    def close(self):

        self.lmdb_env.close()




class LMDBParser_test():

    def __init__(self, lmdb_dir_path, log_dir='/tmp', keep=32):

        # 1. Path to directory containing data.mdb
        # 2. keep which will take 1000 records from lmdb file into CPU memory

        self.lmdb_dir_path = lmdb_dir_path
        self.keep = keep
        self.datum = Datum()

        self.io = EmbeddingIO(log_dir, log_type='lmdb-parsing-{0}'.format(lmdb_dir_path.split('/')[-1]))

        if not os.path.exists(self.lmdb_dir_path):
            self.io.print_error('Could not find LMDB path {0}'.format(self.lmdb_dir_path))
            self.init = False
            return

        try:

            self.lmdb_env = lmdb.open(self.lmdb_dir_path, readonly=True, lock=False)
            self.lmdb_txn = self.lmdb_env.begin()
            self.read_label = self.lmdb_txn.get
            self.lmdb_cursor = self.lmdb_txn.cursor()
            self.lmdb_cursor.first()

        #self.io.print_info('Setting LMDB file : {0} '.format(self.lmdb_dir_path))

        except Exception as e:

            self.io.print_error('Error reading LDMB file {0},{1}'.format(self.lmdb_dir_path, e))
            self.init = False
            return
        self.read_labels=pickle.load(open('/nas2/praveen/home/rnd-libs/rnd_libs/lib/keras/key_labels', 'rb'))
        self.init = True
    def get_labels(self,ID):
        
        try:
            return self.read_labels[ID]
        except KeyError as e:
            #print "Cannot get label for %s" % ID
            return []
        
        return lab
    # def get_labels(self,ID):
    #     import ipdb; ipdb.set_trace()

    #     while 1:

    #         key, value = self.lmdb_cursor.item()

    #         self.datum.ParseFromString(value)
    #         lab_old = self.datum.label
    #         self.lmdb_cursor.next()
    #         import ipdb; ipdb.set_trace()
        # stream = self.read_label(ID)
        # #key,value=self.lmdb_cursor.item()
        # if stream == None:
        #     return []
        # else:
        #      return pickle.loads(stream)

