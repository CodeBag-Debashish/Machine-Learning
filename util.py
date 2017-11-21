from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

import numpy as np
import tensorflow as tf
import matplotlib
import sys
import os
from matplotlib import pyplot as plt
import cv2
from tqdm import tqdm
from random import shuffle


TRAIN_DIR = './train'
TEST_DIR = './test'
IMG_SIZE = 50
LEARNING_RATE = 0.001
MODEL_NAME = 'cs725finalproj-{}-{}.model'.format(LEARNING_RATE,'TF-conv')

def label_img(image):
	label = image.split('.')[-3]
	if label == 'cat':
		return 0
	elif label == 'dog':
		return 1

train_data = []

for image_name in tqdm(os.listdir(TRAIN_DIR)):
	label = label_img(image_name)
	path = os.path.join(TRAIN_DIR,image_name)
	train_data.append([path,label])

shuffle(train_data)
shuffle(train_data)
shuffle(train_data)
shuffle(train_data)
shuffle(train_data)
shuffle(train_data)


cnt = 0
for i in tqdm(train_data):
	if cnt < 2000:
		print(i[0]," ",i[1])
		cnt += 1
	else:
		break



