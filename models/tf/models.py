import os,sys
import numpy as np
from eyelibs.models.tf.googlenet import GoogleNet
from eyelibs.models.tf.squeezenet import SqueezeNet
from eyelibs.models.tf.aesthooglenet import AesthoogleNet

model_dict = {'googlenet':GoogleNet,'squeezenet':SqueezeNet,'aesthooglenet':AesthoogleNet}