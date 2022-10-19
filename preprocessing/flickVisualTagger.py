#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2022
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________

import argparse
import os
import pickle
import sys
import time
import torch
import numpy as np
from tqdm import tqdm
from image_caption_dataset_preprocessing import flickrData
import json
from pathlib import Path
import numpy
from collections import Counter
import sys
from os import popen
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from functools import partial

terminal_rows, terminal_width = popen('stty size', 'r').read().split()
terminal_width = int(terminal_width)
def heading(string):
    print("_"*terminal_width + "\n")
    print("-"*10 + string + "-"*10 + "\n")

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--image-base", default="/storage", help="Path to images.")
command_line_args = parser.parse_args()

classes = []
targets = {}
for line in open(Path('../data/flickr8k.tags.all.txt'), 'r'):
    if len(line.strip().split()) == 0: continue
    if line.strip().split()[0] == '#': continue
    elif line.strip().split()[0] == 'Tags:':
        for kw in line.strip().split()[1:]: classes.append(kw)
    else:
        image = line.strip().split()[0]
        if image not in targets:
            targets[image] = np.zeros(len(classes))
            for i, val in enumerate(line.strip().split()[1:]):
                targets[image][i] = float(val)
            
# classes = set(classes)
# print(classes)

vocab = set()
with open('../data/34_keywords.txt', 'r') as f:
# with open('../data/vpkl_keywords.txt', 'r') as f:
    for keyword in f:
        vocab.add(keyword.strip())

vocab_key = {}
for i, kw in enumerate(vocab): 
    vocab_key[kw] = i

print(len(vocab.intersection(classes)))
ind_to_kw = {}
kw_indices = []

count = 0
vocab_to_classes_key = {}
for i, kw in enumerate(classes):
    if kw in vocab:
        kw_indices.append(i)
        vocab_to_classes_key[count] = i
        count += 1
    ind_to_kw[i] = kw

threshold = 0.6
image_targets = {}
labels_to_images = {}
for image in targets:

    valid_values = targets[image][kw_indices]
    valid_ind = np.argsort(valid_values)[::-1][0:3]

    for ind in valid_ind:
        if valid_values[ind] < threshold: continue
        kw = ind_to_kw[vocab_to_classes_key[ind]] #ind_to_kw[kw_indices[ind]]
        id = vocab_key[kw]

        if image not in image_targets: image_targets[image] = []
        image_targets[image].append(id)

        if id not in labels_to_images: labels_to_images[id] = []
        labels_to_images[id].append(image)
        
np.savez_compressed(
    Path('../data/flickr_visual_tagger_targets.npz'),
    image_targets=image_targets, 
    vocab_key=vocab_key,
    vocab=vocab,
    labels_to_images=labels_to_images
)