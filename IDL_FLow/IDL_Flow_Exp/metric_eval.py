import os
import torch
import numpy as np
import sys

from Utils.context_fid import Context_FID
from Utils.metric_utils import display_scores
from Utils.discriminative_metric import discriminative_score_metrics

iterations = 5
#Change the path to the original data and the generated data as needed
ori_data = np.load('./stock_exp_idl/samples/stock_norm_truth_24_train.npy')
fake_data = np.load('./stock_exp_idl/idl_fake_stock.npy')

context_fid_score = []

for i in range(iterations):
    context_fid = Context_FID(ori_data[:], fake_data[:ori_data.shape[0]])
    context_fid_score.append(context_fid)
    print(f'Iter {i}: ', 'context-fid =', context_fid, '\n')
      
display_scores(context_fid_score)

discriminative_score = []

for i in range(iterations):
    temp_disc, fake_acc, real_acc = discriminative_score_metrics(ori_data[:], fake_data[:ori_data.shape[0]])
    discriminative_score.append(temp_disc)
    print(f'Iter {i}: ', temp_disc, ',', fake_acc, ',', real_acc, '\n')
      
display_scores(discriminative_score)
