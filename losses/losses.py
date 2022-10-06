#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2020
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________

import math
import pickle
import numpy as np
import torch
from .util import *
    
def compute_matchmap_similarity_matrix_loss(
    image_outputs, english_output, english_nframes, negatives, positives, attention, contrastive_loss, 
    margin, simtype, alphas, rank):
    
    loss = 0
    C_aud, C_im = compute_matchmap_similarity_matrix_IA(image_outputs, None, english_output, english_nframes, attention, simtype)
    # print(C_aud, C_im, '\n')
    neg_aud_Cs = []
    neg_im_Cs = []
    for neg_dict in negatives:
        C_neg_aud, _ = compute_matchmap_similarity_matrix_IA(image_outputs, None, neg_dict["english_output"], neg_dict["english_nframes"], attention, simtype)
        neg_aud_Cs.append(C_neg_aud)
        _, C_neg_im = compute_matchmap_similarity_matrix_IA(neg_dict['image'], None, english_output, english_nframes, attention, simtype)
        neg_im_Cs.append(C_neg_im)
        # print(C_neg_aud, C_neg_im, '\n')

    neg_aud_Cs = torch.cat(neg_aud_Cs, dim=1)
    neg_im_Cs = torch.cat(neg_im_Cs, dim=1)

    pos_aud_Cs = []
    pos_im_Cs = []
    for pos_dict in positives:
        C_pos_aud, C_pos_im = compute_matchmap_similarity_matrix_IA(image_outputs, None, pos_dict["english_output"], pos_dict["english_nframes"], attention, simtype)
        pos_aud_Cs.append(C_pos_aud)
        C_pos_aud, C_pos_im = compute_matchmap_similarity_matrix_IA(pos_dict['image'], None, english_output, english_nframes, attention, simtype)
        pos_im_Cs.append(C_pos_im)

    pos_aud_Cs = torch.cat(pos_aud_Cs, dim=1)
    pos_im_Cs = torch.cat(pos_im_Cs, dim=1)

    # anch = torch.cat([C_aud, C_im], dim=0)
    # positives = torch.cat([C_im, C_aud], dim=0)
    # negatives = torch.cat([neg_aud_Cs, neg_im_Cs], dim=0)
    # loss += contrastive_loss(anch, positives, negatives) 

    anch = torch.cat([C_aud, C_aud, C_im, C_im], dim=0)
    positives = torch.cat([C_im, pos_aud_Cs, C_aud, pos_im_Cs], dim=0)
    negatives = torch.cat([neg_aud_Cs, neg_im_Cs, neg_im_Cs, neg_aud_Cs], dim=0)
    loss += contrastive_loss(anch, positives, negatives) 

    # loss = 0
    # l1, C_aud, C_im = compute_matchmap_similarity_matrix_IA(image_outputs, None, english_output, english_nframes, torch.ones((1, 1)).to(rank), attention, simtype)
    # # loss += l1.mean()

    # neg_aud_Cs = []
    # neg_im_Cs = []
    # for neg_dict in negatives:
    #     l1, C_neg_aud, C_neg_im = compute_matchmap_similarity_matrix_IA(image_outputs, None, neg_dict["english_output"], neg_dict["english_nframes"], torch.zeros((1, 1)).to(rank), attention, simtype)
    #     # loss += l1.mean()
    #     neg_aud_Cs.append(C_neg_aud)
    #     l1, C_neg_aud, C_neg_im = compute_matchmap_similarity_matrix_IA(neg_dict['image'], None, english_output, english_nframes, torch.zeros((1, 1)).to(rank), attention, simtype)
    #     # loss += l1.mean()
    #     neg_im_Cs.append(C_neg_im)

    # neg_aud_Cs = torch.cat(neg_aud_Cs, dim=1)
    # neg_im_Cs = torch.cat(neg_im_Cs, dim=1)

    # pos_aud_Cs = []
    # pos_im_Cs = []
    # for pos_dict in positives:
    #     l1, C_pos_aud, C_pos_im = compute_matchmap_similarity_matrix_IA(image_outputs, None, pos_dict["english_output"], pos_dict["english_nframes"], torch.ones((1, 1)).to(rank), attention, simtype)
    #     # loss += l1.mean()
    #     pos_aud_Cs.append(C_pos_aud)
    #     l1, C_pos_aud, C_pos_im = compute_matchmap_similarity_matrix_IA(pos_dict['image'], None, english_output, english_nframes, torch.ones((1, 1)).to(rank), attention, simtype)
    #     # loss += l1.mean()
    #     pos_im_Cs.append(C_pos_im)

    # pos_aud_Cs = torch.cat(pos_aud_Cs, dim=1)
    # pos_im_Cs = torch.cat(pos_im_Cs, dim=1)

    # anch = torch.cat([C_aud, C_aud, C_im, C_im], dim=0)
    # positives = torch.cat([pos_aud_Cs, pos_im_Cs, pos_im_Cs, pos_aud_Cs], dim=0)
    # negatives = torch.cat([neg_aud_Cs, neg_im_Cs, neg_im_Cs, neg_aud_Cs], dim=0)
    # loss += contrastive_loss(anch, positives, negatives) 
    # loss += contrastive_loss(C_aud, pos_aud_Cs, neg_aud_Cs)
    # loss += contrastive_loss(C_aud, pos_im_Cs, neg_im_Cs)
    # loss += contrastive_loss(C_im, pos_im_Cs, neg_im_Cs)
    # loss += contrastive_loss(C_im, pos_aud_Cs, neg_aud_Cs)

    # l = contrastive_loss(C_aud, pos_aud_Cs, neg_aud_Cs)
    # print(l)
    # loss += l
    # l = contrastive_loss(C_aud, pos_im_Cs, neg_im_Cs)
    # print(l)
    # loss += l
    # l = contrastive_loss(C_im, pos_im_Cs, neg_im_Cs)
    # print(l)
    # loss += l
    # l = contrastive_loss(C_im, pos_aud_Cs, neg_aud_Cs)
    # print(l, '\n')
    # loss += l
    return loss
