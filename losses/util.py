#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2022
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________

import math
import pickle
import numpy as np
import torch
import torch.nn as nn
from itertools import product
from tqdm import tqdm
from models.util import *
from tqdm import tqdm

def compute_matchmap_similarity_matrix_IA(im, im_mask, audio, frames, attention, simtype='MISA'):
    
    w = im.size(2)
    h = im.size(3)
    im = im.view(im.size(0), im.size(1), -1).transpose(1, 2)
    audio = audio.squeeze(2)

    assert(im.dim() == 3)
    assert(audio.dim() == 3)
    
    n = im.size(0)
    aud_C = []
    im_C = []

    for i in range(n):
        english_context, image_context = attention(im[i, :, :].unsqueeze(0), audio[i, :, :].unsqueeze(0), frames[i])
        aud_C.append(english_context.unsqueeze(0))
        im_C.append(image_context.unsqueeze(0))

    aud_C = torch.cat(aud_C, dim=0) 
    im_C = torch.cat(im_C, dim=0) 

    return aud_C, im_C
    
def compute_large_matchmap_similarity_matrix_IA(im, im_mask, audio, frames, attention, simtype='MISA'):
    
    w = im.size(2)
    h = im.size(3)
    im = im.view(im.size(0), im.size(1), -1).transpose(1, 2)
    audio = audio.squeeze(2)
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    assert(im.dim() == 3)
    assert(audio.dim() == 3)
    
    n = im.size(0)
    S = []
    im_C = []

    for i in range(n):
        this_im_row = []
        for j in range(n):
            this_aud, this_im = attention(im[i, :, :].unsqueeze(0), audio[j, :, :].unsqueeze(0), frames[j])
            score = cos(this_aud, this_im)
            this_im_row.append(score.unsqueeze(1))
        this_im_row = torch.cat(this_im_row, dim=1)
        S.append(this_im_row)

    S = torch.cat(S, dim=0) 

    return S

def compute_matchmap_similarity_score_IA(im, im_mask, audio, frames, attention, simtype='MISA'):
    
    # w = im.size(2)
    # h = im.size(3)
    # im = im.view(im.size(0), im.size(1), -1).transpose(1, 2)
    # audio = audio.squeeze(2)

    assert(im.dim() == 3)
    assert(audio.dim() == 3)
    
    S, C = attention(im, audio, frames)
    C = C.unsqueeze(0)

    return S, C