import os, sys
import numpy as np
from rnd_libs.models.keras.vgg_16 import VGG16
from rnd_libs.models.keras.vgg_16_adaptation import VGG16Adap
from rnd_libs.models.keras.vgg_19 import VGG19
from rnd_libs.models.keras.vgg_19_adaptation import VGG19Adap
from rnd_libs.models.keras.alexnet import AlexNet
from rnd_libs.models.keras.googlenet import GoogleNet
from rnd_libs.models.keras.hashnet import AlexHash
from rnd_libs.models.keras.inception_v3 import InceptionV3
from rnd_libs.models.keras.inception_v3_fc import InceptionV3_FC
from rnd_libs.models.keras.inception_v3_finetune import InceptionV3_finetune
from rnd_libs.models.keras.vgg_19_bn import VGG19_BN
from rnd_libs.models.keras.aesthetics import Aesthetics
from rnd_libs.models.keras.inception_v3_noise import InceptionV3NoiseLayer
from rnd_libs.models.keras.inception_v3_info import InceptionV3InfoLayer
from rnd_libs.models.captions.inception_v3_captions import InceptionV3Captions

model_dict = {'vgg16': VGG16,
                'vgg19_bn': VGG19_BN,
                'vgg19': VGG19,
                'googlenet': GoogleNet,
                'alexnet': AlexNet,
                'vgg16_adap': VGG16Adap,
                'vgg19_adap': VGG19Adap,
                'alex_hash': AlexHash,
                'inception_v3': InceptionV3,
                'aesthetics': Aesthetics,
                'inception_v3_fc': InceptionV3_FC,
                'inception_v3_info': InceptionV3InfoLayer,
                'inception_v3_captions': InceptionV3Captions,
                'inception_v3_finetune': InceptionV3_finetune}
