
import os, sys
import yaml
import logging
import cStringIO
from evaluation import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from IPython.core.debugger import Tracer
#
# TODO : this caffe dependecy should be replaced with another image/url loader
# import caffe
#
#
from collections import defaultdict
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
from rnd_libs.lib.keras.lmdb_parser import LMDBParser_test
from rnd_libs.models.keras.models import model_dict
from keras import backend as K
from sklearn.metrics import roc_curve, auc,average_precision_score
import theano
import theano.tensor as T
#import matplotlib.pyplot as plt
import pickle
logger = logging.getLogger(__name__)
def contains(string,substring):
    if substring not in string:
        return True
    else:
        return False

roundoff = lambda x: round(x*100,3)
# from rnd_libs.lib.s3_connector import S3Connector
def power_mean(input,p,along_axis):
    #return T.max(x,axis=1) * (1.0**p)
    x=T.reshape(input, (input.shape[0]*input.shape[1], input.shape[2]*input.shape[3]))
    output = T.sum(T.nnet.softmax(x * p) * x , axis=along_axis)
    return T.reshape(output,(input.shape[0],input.shape[1]))

def numpy_tensor_mean(input,p,along_axis):
    #return T.max(x,axis=1) * (1.0**p)
    x=np.reshape(input, (input.shape[0]*input.shape[1], input.shape[2]*input.shape[3]))
    output = np.max(x , axis=along_axis)
    return [np.reshape(output,(input.shape[0],input.shape[1]))]

input=T.tensor4()
p=T.scalar()

f=theano.function([input,p],[power_mean(input,p,1)])

class TheanoTester():

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
        self.softpooling_func=f
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
        self.test_lmdb = LMDBParser_test(self.cfgs['test_file'], log_dir=self.cfgs['log_dir'])
        self.resize_test=self.cfgs['resize_test']
        # self.load_synset_file()
        # self.test_lmdb = LMDBParser(self.cfgs['test_file'], log_dir=self.cfgs['log_dir'])
        # self.test_lmdb.set_params_test(self.concepts)
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
        #import ipdb; ipdb.set_trace()
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
            #import ipdb; ipdb.set_trace()
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
    #to be used in 
    def get_images_frVisualization(self):
        self.load_synset_file()
        GndTr=dict()
        if not self.init:
            return
        indexes=dict()
        self.test_images = self.io.read_images(self.cfgs['test_file_list'])

        #filterout images from getty
        self.io.print_info('Removing getty scrapped images from validation data')

        filter_getty=[contains(img,'getty-concept-sourcing') for img in self.test_images]
        ind=np.where(np.array(filter_getty)==True)[0]
        test_images=np.array(self.test_images)[ind]
        #---------------

        for image in test_images:
            im_basename = os.path.basename(image)
            im_basename,img_ext = os.path.splitext(im_basename)
            
            g=self.test_lmdb.get_labels(im_basename)
            concepts=np.array(self.concepts)
            index=np.where(g==1)[0]
            GndTr[im_basename] = concepts[index]
            indexes[im_basename]=index
 

        return (test_images,GndTr,indexes)

    def visualize_image(self,beta,pixels):
        self.beta=beta

        self.load_synset_file()



            
        flag, predictions, confidence, t = self.classify_images(pixels)



            #file_paths = [os.path.join(self.cfgs['results_dir'], 'visualization_',str(self.beta)+'_'+im + save_ext) for im in basenames]
    def run(self):

        self.load_synset_file()

        if not self.init:
            return

        self.test_images = self.io.read_images(self.cfgs['test_file_list'])
        image_batches = [self.test_images[i:i + self.cfgs['batch_size']] for i in range(0, len(self.test_images), self.cfgs['batch_size'])]

        self.io.print_info('{0} images split into {1} Batches of {2}'.format(len(self.test_images), len(image_batches), self.cfgs['batch_size']))
        for beta in self.cfgs['betas']:
            self.beta=beta
            for idx, images in enumerate(image_batches):

                if not self.cfgs['force']:

                    flag = self.check_jsons(images)

                    if flag:
                        self.io.print_warning('Skipping batch {0} of {1}'.format(idx, len(image_batches)))
                        continue

                pixels, basenames,dummy = self.fetch_images(images)
                dummy=[]
                
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

                file_paths = [os.path.join(self.cfgs['results_dir'], str(self.beta)+'_'+im + save_ext) for im in basenames]
                
                if not self.raw_predictions:
                    to_save = [dict({'external': p}) for p in predictions_pc]
                else:
                    to_save = predictions_pc
                #import ipdb; ipdb.set_trace()
                
                #to_save = [dict({'external': p}) for p in predictions_pc]
                #to_save=predictions_pc
                #to_save_id = [dict({'external': id_}) for id_ in ids]
                assert(len(file_paths) == len(to_save))

                self.io.print_info('Done batch {0} of {1} in {2} secs'.format(idx, len(image_batches), t))
                #import ipdb; ipdb.set_trace()
                #import ipdb; ipdb.set_trace()
                f = map(self.io.save_good_labels, file_paths, to_save)
            #f = map(self.io.save_good_labels, file_paths, to_save_id)
    def check_jsons(self, images):

        basenames = []

        for image in images:

            im_basename = os.path.basename(image)
            im_basename, _ = os.path.splitext(im_basename)

            basenames.append(im_basename)

        flags = [os.path.exists(os.path.join(self.cfgs['results_dir'], b + '.json')) for b in basenames]

        return np.all(flags)

    def compare_various_betas(self):

        betas=self.cfgs['betas']
        Area_PR=dict()
        dir_path = os.path.dirname(os.path.realpath(__file__))
        #print dir_path
        self.limit_NoofClasses=None
        self.load_synset_file()
        self.load_synset_file_training()
        self.imgHist_Concepts_inTraindata=[(np.where(np.array(self.concepts)==i)[0][0]) for i in self.trained_concepts]

        #import lmdb
        #lmdb_env = lmdb.open("./keywording_test_images_labels", readonly=True, lock=False)
        
        # lmdb_txn = lmdb_env.begin()
        # read_label = lmdb_txn.get
        # label1test=self.test_lmdb.get_labels(read_label,'10085536')
        # label2test=self.test_lmdb.get_labels(read_label,'10085536')
        #self.imgname_labels=self.test_lmdb.create_keys_labels_dict()
        var = dict()
        for beta in betas:
            var.update(self.eval_usinglmdb(str(beta)))
        
        #import ipdb; ipdb.set_trace()
        if self.limit_NoofClasses==None:
            df2 = pd.DataFrame(var,index=self.trained_concepts)
        else:
            df2 = pd.DataFrame(var,index=self.trained_concepts[0:self.limit_NoofClasses])
        
        df2.to_csv(dir_path + '/results/' + self.cfgs['results_dir'].split('/')[-2] + '.csv')
        #import ipdb; ipdb.set_trace()







        
        # mAP=[]
        # mRec90=[]
        # mRec80=[]
        # max_f1=[]
        # for beta in betas:
        #     vAP,v90,v80,vf1=Area_PR[beta]
        #     mAP.append(vAP)
        #     mRec90.append(v90)
        #     mRec80.append(v80)
        #     max_f1.append(vf1)
        #     #print ("mAP:%.4f,mean Recall:%.4f,beta:%s" % (mAP,mRec,str(beta)))
        # #import ipdb; ipdb.set_trace()
        # #print(*betas, sep='\t')
        # print('\t'.join([str(x) for x in betas]))
        # #print(*mRec80, sep='\t')
        # print('\t'.join([str(round(x*100,3)) for x in mRec80]))
        # #print(*mRec90, sep='\t')
        # print('\t'.join([str(round(x*100,3)) for x in mRec90]))
        # print('\t'.join([str(round(x*100,3)) for x in mAP]))
        # print('\t'.join([str(round(x*100,3)) for x in max_f1]))
        #print(*mAP, sep='\t')
        #pickle.dump(Area_PR,open('/nas2/praveen/home/rnd-libs/rnd_libs/lib/keras/result_Area_PR','wb'))

            
     # def generate_csv_file(self):

     #    betas=self.cfgs['betas']
     #    Area_PR=dict()
        

     #    #import lmdb
     #    #lmdb_env = lmdb.open("./keywording_test_images_labels", readonly=True, lock=False)
     #    #import ipdb; ipdb.set_trace()
     #    # lmdb_txn = lmdb_env.begin()
     #    # read_label = lmdb_txn.get
     #    # label1test=self.test_lmdb.get_labels(read_label,'10085536')
     #    # label2test=self.test_lmdb.get_labels(read_label,'10085536')
     #    #self.imgname_labels=self.test_lmdb.create_keys_labels_dict()

     #    for beta in betas:
     #        Area_PR[beta]=self.eval_usinglmdb(str(beta))
        
     #    mAP=[]
     #    mRec90=[]
     #    mRec80=[]
     #    max_f1=[]
     #    detections=[]
     #    for beta in betas:
     #        vAP,v90,v80,max_f1,detections=Area_PR[beta]
     #        mAP.append(vAP)
     #        mRec90.append(v90)
     #        mRec80.append(v80)
            #print ("mAP:%.4f,mean Recall:%.4f,beta:%s" % (mAP,mRec,str(beta)))
        #import ipdb; ipdb.set_trace()
        #print(*betas, sep='\t')
    
        #import ipdb; ipdb.set_trace()
       
    # def compare_various_betas(self):
    #     import ipdb; ipdb.set_trace()
    #     betas=self.cfgs['betas']
    #     for beta in betas:
    #         self.eval_usinglmdb(str(beta)
    def eval_usinglmdb(self,beta):
        
        
        #import ipdb; ipdb.set_trace()
        #import ipdb; ipdb.set_trace()

        self.test_images = self.io.read_images(self.cfgs['test_file_list'])

        self.predictions = {}
        self.gt = {}
        valid_keys=[]
        notgoodGTimgs=[]
        ImgsSmallNumofTags=[]
        #import ipdb; ipdb.set_trace()

        for idx, image in enumerate(self.test_images):

            if idx % 1000 == 0:
                self.io.print_info('Reading {0}/{1}'.format(idx, len(self.test_images)))

            im_basename = os.path.basename(image)
            im_basename,img_ext = os.path.splitext(im_basename)
            
            #result_json = os.path.join(self.cfgs['results_dir'], im_basename + '.json')
            result_json = os.path.join(self.cfgs['results_dir'], beta+'_'+im_basename + '.npz')
            #gt_json = os.path.join(self.cfgs['gt_dir'], im_basename + '.json')
            
            if os.path.exists(result_json):
                #p, s,ids = self.io.read_prediction_json_withIDs(result_json)
                p = np.load(result_json)['features']

            else:
                p=[]
            #g = self.io.read_vector(gt_json)
            #import ipdb; ipdb.set_trace()
            g=self.test_lmdb.get_labels(im_basename)
            #print np.sum(g)
            #print p.shape
            if len(p) == 0:
                print ("images with no prediction generated: %s") %im_basename
                continue


            if len(g) == 0:
                #import ipdb; ipdb.set_trace()
                print ("images with no GT data: %s") %im_basename
                notgoodGTimgs.append(im_basename)
                continue
            if len(g)<7735 and len(g) != 0:
                print ("images with small number of tags: %s") %im_basename
                ImgsSmallNumofTags.append(im_basename)
                continue
            #g_filtered=self.parse_gndTruth_withFilter(ids,g)

            # if p is [] or g is []:
            #     continue
            
            self.predictions[im_basename] = p
            
            self.gt[im_basename] = g
            valid_keys.append(im_basename)
            # if idx>1000:
            #     break
        print 'Number of images: with no GT %d and with small no of tags %d' % (len(notgoodGTimgs),len(ImgsSmallNumofTags))
        #import ipdb; ipdb.set_trace()
        

        y_test=np.zeros((len(valid_keys),len(self.concepts)),dtype='u1')
        y_score=np.zeros((len(valid_keys),len(self.concepts)))
        


        for i,key in enumerate(valid_keys):
            y_test[i]=self.gt[key]
            y_score[i]=self.predictions[key]
        
        gt=dict()
        pred=defaultdict(dict)
        
        for i,concept in enumerate(self.concepts):
            imgname_ind=np.where(y_test[i]==1)
            np_array_photo_ids=np.array(valid_keys)[imgname_ind]
            gt[concept]=list(np_array_photo_ids)
            #pred[concept]={key: value for key, value in self.predictions.items() if key in list(np_array_photo_ids)}
            # for key in list(np_array_photo_ids):
            #     pred[concept].update({key: self.predictions[key]})
        
        
        # pairof_photoid_predictions=self.predictions.items()
        # for photo_id, predictions in pairof_photoid_predictions:
        #     for concept, score in zip(self.concepts,predictions):
        #         pred[concept][photo_id] = score
        # #import ipdb; ipdb.set_trace()
        # thresholds=get_thresholds(pred, self.concepts, gt)
        # detections=get_detections_per_concept(pred, self.concepts, gt, thresholds)
        

        #class_histogram=y_test.sum(axis=0)
        #import ipdb; ipdb.set_trace()

        if self.cfgs['percentate_of_classesFrEval'] != None or self.cfgs['percentate_of_classesFrEval'] < 1.0:
            use_subsetofClsFrEval=True
            n_classes=len(self.concepts)

            limit_NoofClasses=int(self.cfgs['percentate_of_classesFrEval'] * n_classes)

            subset_indices_FrEval=self.imgHist_Concepts_inTraindata[:limit_NoofClasses]
            self.limit_NoofClasses=limit_NoofClasses
            #self.concepts=np.array(self.concepts)[indices]

            #y_test_subset=np.zeros((len(valid_keys),len(self.concepts)),dtype='u1')
            #y_score_subset=np.zeros((len(valid_keys),len(self.concepts)))

            #for i,key in enumerate(valid_keys):
            #    y_test_subset[i]=self.gt[key][indices]
            #    y_score_subset[i]=self.predictions[key][indices]
            #print "Using %d out of %d rare number of classes for evaluations" % (limit_NoofClasses,n_classes)
        else:
            use_subsetofClsFrEval=False
            subset_indices_FrEval=self.imgHist_Concepts_inTraindata
            #subset_indices_FrEval=range(len(self.concepts))

        #if use_subsetofClsFrEval:
        #    y_test=y_test_subset
        #    y_score=y_score_subset

        #plt.plot(np.sort(temp)[::-1])

        #plt.savefig('/nas2/praveen/home/rnd-libs/rnd_libs/lib/keras/intermediate/class_histogram')
        
        #fpr = dict()
        #tpr = dict()
        recAt90 = []
        recAt70 = []
        ap=[]
        names=['ap','recAt70','recAt90']
        #max_f1=[]
        
        for ind in subset_indices_FrEval:
            #import ipdb; ipdb.set_trace()
            # if y_test[:, i].sum()==0.0:
            #     continue
            #fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            
            ap.append(roundoff(average_precision_score(y_test[:, ind], y_score[:, ind]))) 
            recAt90.append(roundoff(recall_at_specified_precision(y_test[:, ind], y_score[:, ind], specified_precision=0.9)))
            recAt70.append(roundoff(recall_at_specified_precision(y_test[:, ind], y_score[:, ind], specified_precision=0.7)))
            #max_f1[self.concepts[ind]] = max_f1_score(y_test[:, ind], y_score[:, ind])


        #import ipdb; ipdb.set_trace()
        #return (np.array(ap.values()).mean(), np.array(recAt90.values()).mean(), np.array(recAt80.values()).mean())
        #max_f1=[0.0]
        #import ipdb; ipdb.set_trace()
        returndict=dict()

        for name in names:
            returndict[beta+'_'+name]=locals()[name]
        return (returndict)
        #import ipdb; ipdb.set_trace()
        # import ipdb; ipdb.set_trace()  
        # list_thres=[self.cfgs['global_threshold']]      
        # for t in list_thres:
        #     self.compute_precision_recall(t)
    def eval(self):
        
        self.load_synset_file()

        self.test_images = self.io.read_images(self.cfgs['test_file_list'])

        self.predictions = {}
        self.gt = {}

        for idx, image in enumerate(self.test_images):

            if idx % 1000 == 0:
                self.io.print_info('Reading {0}/{1}'.format(idx, len(self.test_images)))

            im_basename = os.path.basename(image)
            im_basename,_ = os.path.splitext(im_basename)

            result_json = os.path.join(self.cfgs['results_dir'], im_basename + '.json')
            gt_json = os.path.join(self.cfgs['gt_dir'], im_basename + '.json')
            
            p, s = self.io.read_prediction_json(result_json)
            g = self.io.read_vector(gt_json)

            if p is [] or g is []:
                continue
            
            self.predictions[im_basename] = zip(p, s)
            self.gt[im_basename] = list(set(g).intersection(self.concepts))

        for t in [self.cfgs['thresholds']]:
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
    def load_synset_file_training(self):

        pfile = open(self.cfgs['synset_file_training'], 'r')
        concepts = pfile.readlines()
        pfile.close()

        self.trained_concepts = [p.strip() for p in concepts]

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

        concepts = [(self.concepts[idx], float(p[idx]),idx) for idx in sort_idx[:self.cfgs['top_k']['test']]]
        #import ipdb; ipdb.set_trace()
        return concepts
    def parse_gndTruth_withFilter(self, ids,p):
        
        p=np.array(p)
        concepts=np.array(self.concepts)
        filtered_gt=p[ids]
        filtered_concepts=concepts[ids]


        gt_ids=np.where(filtered_gt==1)[0]

        #sort_idx = np.argsort(-p)
         
        
        
        #import ipdb; ipdb.set_trace()
        return filtered_concepts[gt_ids]
    def parse_gndTruth(self, p):
        #import ipdb; ipdb.set_trace()
        gt=np.array(p)
        concepts=np.array(self.concepts)
        #filtered_gt=p[ids]
        #filtered_concepts=concepts[ids]


        gt_ids=np.where(gt==1)[0]

        #sort_idx = np.argsort(-p)
  
      
        #import ipdb; ipdb.set_trace()
        return concepts[gt_ids]
    def classify_images(self, images):

        confidence = [None]

        if images == []:
            return False, [(), ()], 0.0
        try:

            tstart = time.time()
            
            (predictions, features) = self.inference(images)
            #import ipdb; ipdb.set_trace()
            predictions=f(predictions,self.beta)[0]
            #import ipdb; ipdb.set_trace()
            #pickle.dump(predictions,open('predictions','wb'))
            #test=f(predictions,self.beta)[0]
            #predictions=numpy_tensor_mean(predictions,self.beta,1)[0]
            
            #import ipdb; ipdb.set_trace()
            # if len(predictions.shape) == 4:
            #     predictions=predictions[:,:,0,0]
            #import ipdb; ipdb.set_trace()   
            if self.label_vectors is not None:
                predictions /= np.linalg.norm(predictions, axis=1)[:, np.newaxis]
                predictions = np.dot(predictions, self.label_vectors)

            if self.confidence_model is not None:
                confidence = [('Confidence', p) for p in self.confidence_model.predict(features)]

            tend = time.time()

        except Exception as err:
            self.io.print_error('Error processing image, {0}'.format(err))
            return False, [(), ()], None, 0.0
           
        # if not self.raw_predictions:
        #     results = [self.parse_predictions(p)[1] for p in predictions] , True, [self.parse_predictions(p)[0] for p in predictions], confidence, '{0:.4f}'.format((tend - tstart))
        # else:
        #     results = True, predictions, confidence, '{0:.4f}'.format((tend - tstart))
        #results = True, [self.parse_predictions(p) for p in predictions], confidence, '{0:.4f}'.format((tend - tstart))
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
            if self.resize_test:
                #Tracer()()
                #print image.size
                image = cv2.resize(image, (self.im_h, self.im_w)).astype(np.float32)
            else:
                print 'no resize'
                image=image.astype(np.float32)
        except Exception as e:
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
            
            image_file_name=image_file_name.replace('/nas/','/nas2/')

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
            #orig_image=image.astype('u1')
            
            image = self.scale_image(image)

            if image is None:
                continue

            if self.verbose:
                self.io.print_info('Processing {0} {1}'.format(image_file_name, image.shape))

            images.append(image)

            basenames.append(im_basename)

        tend = time.time()

        self.io.print_info('Fetching {0} images took {1:.4f} secs'.format(len(image_file_names), tend - tstart))

        #return ([np.array(images)], basenames,orig_images)
        return ([np.array(images)], basenames)

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
