# Copyright 2017 Calico LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
#
# Modifications copyright 2025 by Ada Zhang
#
# =========================================================================

from __future__ import print_function
from optparse import OptionParser

import json
import os
import shutil
import sys
import time
from tqdm import tqdm

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import mixed_precision
if tf.__version__[0] == '1':
    tf.compat.v1.enable_eager_execution()

import sys
sys.path.insert(0,'./')
from basenji import dataset
from basenji import seqnn
from basenji import trainer

import pickle

"""
predict_test_set.py

Output trained Basenji model predictions on test set
"""

################################################################################
# main
################################################################################
def main():
    usage = 'usage: <model_dir> <data1_dir> ...'
    
    parser = OptionParser(usage)
    (options, args) = parser.parse_args()

    if len(args) < 2:
        parser.error('Must provide parameters and data directory.')
    else:
        model_dir = args[0]
        params_file = f'{model_dir}/params.json'
        # inds_file = f'{model_dir}/inds.pkl'
        data_dir = args[1]

    # read model parameters
    with open(params_file) as params_open:
        params = json.load(params_open)
    params_model = params['model']
    params_train = params['train']
    
    # # import indices
    # if os.path.exists(inds_file):
    #     with open(inds_file,'rb') as inds_open:
    #         data_inds = pickle.load(inds_open)
    # else:
    #     data_inds = None

    # load test data
    test_data = dataset.SeqDataset(data_dir,
                                   split_label='test',
                                   batch_size=1,
                                   mode='eval',
                                   tfr_pattern=None)

    # initialize model
    seqnn_model = seqnn.SeqNN(params_model)

    # restore
    if os.path.exists(f'{model_dir}/model_best.h5'):
        seqnn_model.restore(f'{model_dir}/model_best.h5')
    elif os.path.exists(f'{model_dir}/model_human.h5'):
        seqnn_model.restore(f'{model_dir}/model_human.h5')
        

    # initialize numpy matrices to hold data
    pred = np.zeros((test_data.num_seqs, test_data.target_length,params_model['head_human']['units']))
    gt = np.zeros_like(pred)
    for i, (x,y) in enumerate(tqdm(iter(test_data.dataset),total=test_data.num_seqs)):
        # if data_inds is not None:
        #     y = tf.gather(y,indices=data_inds,axis=2)

        pred_i = seqnn_model.model(x, training=False)
        pred[i] = pred_i.numpy()
        gt[i] = y.numpy()

    with open(f'{model_dir}/model_best_test_set_gt.pickle','wb') as f:
        pickle.dump(gt,f)
    with open(f'{model_dir}/model_best_test_set_pred.pickle','wb') as f:
        pickle.dump(pred,f)
        
################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
