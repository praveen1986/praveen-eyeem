import os
import sys
import cPickle
import json
import csv
import logging
import urllib
import numpy as np
from termcolor import colored
from time import strftime, localtime

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class EmbeddingIO():

    def __init__(self,logging_dir,log_type='training',quite=False,reduced=False):

        self.quite = quite
        self.logging_dir = logging_dir

        if not self.logging_dir == None:
            ts = self.get_local_time()
            ts = '-'.join(ts.split(' '))
            handler = logging.FileHandler(os.path.join(self.logging_dir,ts+'-'+log_type+'.log'))
            handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        #if

    #def

    def print_info(self,info_string,quite=False):

        info = '[{0}][INFO] {1}'.format(self.get_local_time(),info_string)

        if not ( self.quite or quite ):
            print colored(info,'green')
        #if

        logger.info(info)

    #def

    def print_warning(self,warning_string):

        warning = '[{0}][WARNING] {1}'.format(self.get_local_time(),warning_string)

        if not self.quite:
            print colored(warning,'blue')
        #if

        logger.warning(warning)

        #def

    def print_error(self,error_string):

        error = '[{0}][ERROR] {1}'.format(self.get_local_time(),error_string)

        if not self.quite:
            print colored(error,'red')
        #if

        logger.error(error)

    #def

    def get_local_time(self):

        return strftime("%d %b %Y %Hh%Mm%Ss", localtime())

    #def

    def begin_logging(self):

        log_time = self.get_local_time()
        self.print_info('Begin logging {0}\n'.format(log_time))

    #def

    def end_logging(self):

        log_time = self.get_local_time()
        self.print_info('End logging {0}\n'.format(log_time))

    #def

    def print_results(self,prefix,concepts,scores,quite=False):

        for c,s in zip(concepts,scores):
            self.print_info('[{0}] {1} {2}'.format(prefix,c,s),quite)
        #for

    #def

    def read_hedged_labels(self,path_bet_file):

        pfile = open(path_bet_file,'r')
        bet = cPickle.load(pfile)
        pfile.close()

        all_labels = bet['words']

        return all_labels

    #def

    def read_wn_labels(self,path_bet_file,path_synset_file):

        pfile = open(path_bet_file,'r')
        bet = cPickle.load(pfile)
        pfile.close()

        all_labels = bet['words']

        pfile = open(path_synset_file,'r')
        leaf_labels = pfile.readlines()
        leaf_labels = [ l.split(' ')[1] for l in leaf_labels ]
        pfile.close()

        info = 'Found {0} leaf labels and {1} internal labels'.format(len(leaf_labels),len(all_labels)-len(leaf_labels))
        self.print_info(info)

        return leaf_labels, all_labels

    #def

    def read_synset_file(self,path_synset_file):


        pfile = open(path_synset_file,'r')
        labels = pfile.readlines()
        pfile.close()

        labels = [ l.strip() for l in labels]

        self.print_info('Found {0} labels in {1}'.format(len(labels),path_synset_file))

        return labels

    #def

    def read_getty_labels(self,path_getty_file,stop_count,allowed_types):

        pfile = open(path_getty_file,'r')
        reader = csv.reader(pfile)

        all_getty = [ (r[0],r[1],r[2]) for r in reader ]
        pfile.close()

        getty_labels = []

        for getty, fcount, ftype in all_getty[1:] :

            if int(fcount) < stop_count :
                break
            #if

            if not allowed_types == ['all']:
                if ftype in allowed_types:
                    getty_labels.append(getty.lower())
                #if
            else:
                getty_labels.append(getty.lower())
            #if

        #for

        self.print_info('Found {0} getty labels'.format(len(getty_labels)))

        return getty_labels

    #def

    def read_images(self,image_file_list):

        pfile = open(image_file_list,'r')
        image_names = pfile.readlines()
        pfile.close()

        image_names = [ i.strip() for i in image_names ]

        return image_names

    #def

    def read_vector(self,file_path):

        if not os.path.exists(file_path):
            self.print_warning('Skipping {0}, no such file'.format(file_path))
            return []
        #if

        if file_path.endswith('.pk'):

            dense_pred = []

            try:
                pfile = open(file_path,'r')
                d = cPickle.load(pfile)
                pfile.close()

                all_pred = d['dense_pred']

                for pred in all_pred:
                    dense_pred.extend(pred)
                #for

            except Exception as e:
                self.print_warning('Could not read {0},{1}'.format(file_path,e))
            #try

            return dense_pred

        #if

        if file_path.endswith('.json'):

            try:

                pfile = open(file_path,'r')
                d = json.load(pfile)
                pfile.close()

                return [ s.lstrip().lower() for s in d['external'] ]

            except Exception as e:
                self.print_warning('Could not read {0},{1}'.format(file_path,e))
                return []
            #try

        #if

        if file_path.endswith('.npz'):

            dense_pred = []

            try:

                pfile = open(file_path,'r')
                d = np.load(pfile)

                dense_pred = d['global_features']

                pfile.close()

            except Exception as e:
                self.print_warning('Could not read {0},{1}'.format(file_path,e))
                return []
            #try

            return dense_pred

        #if

        self.print_warning('Extention not recognized or reading protocol not defined for this file,{0}'.format(file_path))

        return []

    #def

    def read_prediction_json(self,json_file):

        try:

            pfile = open(json_file,'r')
            d = json.load(pfile)
            pfile.close()

            labels = [ s[0] for s in d['external'] ]
            scores = [ float(s[1]) for s in d['external'] ]

            return labels, scores

        except Exception as e:
            self.print_warning('Could not read {0},{1}'.format(json_file,e))
            return [],[]
        #try

    def read_prediction_json_withIDs(self,json_file):

        try:

            pfile = open(json_file,'r')
            d = json.load(pfile)
            pfile.close()

            labels = [ s[0] for s in d['external'] ]
            scores = [ float(s[1]) for s in d['external'] ]
            ids = [ int(s[2]) for s in d['external'] ]
            return labels, scores,ids

        except Exception as e:
            self.print_warning('Could not read {0},{1}'.format(json_file,e))
            return [],[]
    #def

    def read_wn_vectors(self,image_names, location_dir, suffix):

        vectors = []

        for idx, image in enumerate(image_names):

            basename = os.path.basename(image)
            basename, ext = os.path.splitext(basename)

            if idx % 1000 == 0:
                self.print_info('Reading {0} of {1} images '.format(idx,len(image_names)))
            #if

            vec_path = os.path.join(location_dir,basename+suffix)
            vectors.append(self.read_vector(vec_path))

        #for

        return vectors

    #def

    def read_getty_vectors(self,image_names,location_dir,suffix):

        vectors = []

        for idx, image in enumerate(image_names):

            basename = os.path.basename(image)
            basename, ext = os.path.splitext(basename)

            if idx % 1000 == 0:
                self.print_info('Reading {0} of {1} images '.format(idx,len(image_names)))
            #if

            vec_path = os.path.join(location_dir,basename+suffix)
            vectors.append(self.read_vector(vec_path))

        #for

        return vectors

    #def

    def save_npz_file(self,dense,getty,npz_file_path):

        np.savez(npz_file_path,dense=dense,getty=getty)

    #def

    def read_npz_file(self,npz_file_path):

        pfile = open(npz_file_path,'r')
        d = np.load(pfile)

        D = d['dense']
        G = d['getty']

        pfile.close()

        return D,G

    #def

    def save_weight_matrix(self,W,weight_file):

        np.savez(weight_file,weight=W)

    #def

    def save_emb_file(self,emb,emb_file):

        pfile = open(emb_file,'wb')
        cPickle.dump(emb,pfile)
        pfile.close()

    #def

    # def read_npz_file(self, npz_file, tag):

    #     pfile = open(npz_file, 'r')
    #     info = np.load(pfile)
    #     info = info[tag]
    #     pfile.close()

    #     return info
    def read_npz_file(self, npz_file, tags):

        if type(tags) == str:
            tags = [tags]

        pfile = open(npz_file, 'r')
        info = np.load(pfile)
        info = [info[tag] for tag in tags]
        pfile.close()

        if len(tags) == 1:
            return info[0].item()
        else:
            return info

    def read_emb_file(emb_file):

        pfile = open(emb_file,'rb')
        emb = cPickle.load(pfile)
        pfile.close()

        return emb

    #def

    def read_weight_matrix(self,weight_file):

        pfile = open(weight_file,'r')

        W = np.load(pfile)
        W = W['weight']
        pfile.close()

        return W

    #def

    def save_threshold_file(self,filepath, thresholds, black_list):

        np.savez(filepath,thresholds=thresholds,black_list=black_list)

    #def

    def read_threshold_file(self,filepath):

        pfile = open(filepath,'r')
        d = np.load(pfile)

        threshold = d['thresholds']
        black_list = d['black_list']

        pfile.close()

        return threshold, black_list

    #def

    def save_good_labels(self,save_file,d):

        filename, ext = os.path.splitext(save_file)

        if ext == '.json':
            pfile = open(save_file,'w')
            json.dump(d,pfile)
            pfile.close()
        elif ext == '.npz':
            pfile = open(save_file,'w')
            np.savez(pfile,features=d)
            pfile.close()
        else:
            self.print_error('Ext not understood {}, try one of .json or .npz'.format(ext))
        #if

    #def

    def write_all_emb(self,emb_file,W):

        np.savez(emb_file,all_svms=W)

    #def

    def read_all_emb(self,labels,method,dir_path):

        embs = {}

        # TODO : Prepare all emb path outside this function, parent function

        if os.path.isdir(dir_path):

            for idx,l in enumerate(labels):

                if idx % 1000 == 0:
                    self.print_info('{1}/{2} Reading embedding for {0}'.format(l,idx,len(labels)))
                #if

                emb_file_path = os.path.join(dir_path,method+'-'+l+'.pkl')

                if not os.path.exists(emb_file_path):
                    self.print_warning('Skipping {0}, no such file or directory'.format(emb_file_path))
                    continue
                #if

                embs[l] = self.read_emb_file(emb_file_path)

            #for

        else:

            all_emb_file = dir_path
            pfile = open(all_emb_file,'r')
            npzfile = np.load(pfile)
            d = npzfile['all_svms']
            embs = d.item()
            pfile.close()

            self.print_info('Read {1} per concept embeddings from {0}'.format(all_emb_file,len(embs.keys())))

        #if

        # embs is dictionary of <concept_name>: <linear embedding>
        return embs

    #def

#class
