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
import tqdm
from natsort import natsorted
import glob

import numpy as np
import tensorflow as tf
if tf.__version__[0] == '1':
    tf.compat.v1.enable_eager_execution()
import dataset

import pyfaidx
import tensorflow_hub
import kipoiseq
from kipoiseq import Interval

import pickle

"""
predict_test_set.py

Output trained Basenji model predictions on test set
"""

# =========================================================================
# FastaStringExtractor and Enformer classes and the one_hot_encode function were
# pulled from https://github.com/google-deepmind/deepmind-research/blob/master/enformer/enformer-usage.ipynb

# Copyright 2021 DeepMind Technologies Limited

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

class FastaStringExtractor:
    
    def __init__(self, fasta_file):
        self.fasta = pyfaidx.Fasta(fasta_file)
        self._chromosome_sizes = {k: len(v) for k, v in self.fasta.items()}

    def extract(self, interval: Interval, **kwargs) -> str:
        # Truncate interval if it extends beyond the chromosome lengths.
        chromosome_length = self._chromosome_sizes[interval.chrom]
        trimmed_interval = Interval(interval.chrom,
                                    max(interval.start, 0),
                                    min(interval.end, chromosome_length),
                                    )
        # pyfaidx wants a 1-based interval
        sequence = str(self.fasta.get_seq(trimmed_interval.chrom,
                                          trimmed_interval.start + 1,
                                          trimmed_interval.stop).seq).upper()
        # Fill truncated values with N's.
        pad_upstream = 'N' * max(-interval.start, 0)
        pad_downstream = 'N' * max(interval.end - chromosome_length, 0)
        return pad_upstream + sequence + pad_downstream

    def close(self):
        return self.fasta.close()

# @title `Enformer`, `EnformerScoreVariantsNormalized`, `EnformerScoreVariantsPCANormalized`,
SEQUENCE_LENGTH = 393216

class Enformer:

    def __init__(self, tfhub_url):
        self._model = tensorflow_hub.load(tfhub_url).model

    def predict_on_batch(self, inputs):
        predictions = self._model.predict_on_batch(inputs)
        return {k: v.numpy() for k, v in predictions.items()}

    @tf.function
    def contribution_input_grad(self, input_sequence,
                                target_mask, output_head='human'):
        input_sequence = input_sequence[tf.newaxis]

        target_mask_mass = tf.reduce_sum(target_mask)
        with tf.GradientTape() as tape:
            tape.watch(input_sequence)
            prediction = tf.reduce_sum(
                target_mask[tf.newaxis] *
                self._model.predict_on_batch(input_sequence)[output_head]) / target_mask_mass

        input_grad = tape.gradient(prediction, input_sequence) * input_sequence
        input_grad = tf.squeeze(input_grad, axis=0)
        return tf.reduce_sum(input_grad, axis=-1)

def one_hot_encode(sequence):
    return kipoiseq.transforms.functional.one_hot_dna(sequence).astype(np.float32)

# =========================================================================
# augment_data and get_model_output were written by heavily referencing
# augmentation code available here: https://github.com/calico/basenji/blob/master/basenji/layers.py
#
# Copyright 2019 Calico LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 
# Modifications copyright 2025 by Ada Zhang
# =========================================================================

def augment_data(seq):
    # stochastic reverse compliment
    reverse = np.random.choice([True,False])
    
    if reverse:
        seq = tf.gather(seq, [3, 2, 1, 0], axis=-1) # flips ACGT to compliment
        seq = tf.reverse(seq, axis=[1])          # reverses sequence

    # stochastic shift with constant padding
    input_shape = seq.shape
    shift_i = np.random.randint(-3,4)
    pad = 0.25 * tf.ones_like(seq[:, 0:tf.abs(shift_i), :])

    def _shift_right(_seq):
        # shift is positive
        sliced_seq = _seq[:, :-shift_i:, :]
        return tf.concat([pad, sliced_seq], axis=1)

    def _shift_left(_seq):
        # shift is negative
        sliced_seq = _seq[:, -shift_i:, :]
        return tf.concat([sliced_seq, pad], axis=1)
    
    if shift_i > 0:
        seq = _shift_right(seq)
    elif shift_i < 0:
        seq = _shift_left(seq)
    seq.set_shape(input_shape)

    return seq, reverse

def get_model_output(x,model):
    x = tf.expand_dims(x,axis=0)
    
    # augment 8 times
    x_aug = []
    reverse = []
    for i in range(8):
        xa,r = augment_data(x)
        x_aug.append(xa)
        reverse.append(r)
        
    x_aug = tf.squeeze(tf.convert_to_tensor(x_aug))
    
    i = 0
    pred = []
    while i < len(x_aug):
        end = min(len(x_aug),i+4)
        pred.append(model.predict_on_batch(x_aug[i:end])['human'])
        i = end
    pred = np.concatenate(pred)
    
    # average predictions across 8 augmentations
    for i in range(8):
        if reverse[i]:
            pred[i] = np.flip(pred[i],axis=0)
    pred = np.mean(pred,axis=0)
        
    return pred
    
################################################################################
# main
################################################################################
def main():
    
    usage = 'usage: <data1_dir> ...'
    
    parser = OptionParser(usage)
    (options, args) = parser.parse_args()

    if len(args) < 1:
        parser.error('Must provide data directory.')
    else:
        data_dir = args[0]

    # load test data
    test_data = dataset.SeqDataset(data_dir,
                                   split_label='test',
                                   batch_size=1,
                                   mode='eval',
                                   tfr_pattern=None)
    # grab sequence info
    seq_info_file = f'{data_dir}/sequences.bed'
    seq_ind = 0
    seq_coords = []
    with open(seq_info_file) as f:
        for line in tqdm.tqdm(f,total=38170):
            chrom, s, e, partition = line.strip('\n').split('\t')
            
            if partition != 'test':
                # training or validation sample, so skip
                continue 
                
            seq_coords.append([chrom,int(s),int(e)])
            seq_ind +=1

    model = Enformer('https://tfhub.dev/deepmind/enformer/1')
    fasta_extractor = FastaStringExtractor('./genome.fa')

    # initialize numpy matrices to hold data
    pred = np.zeros((test_data.num_seqs, test_data.target_length,test_data.num_targets))
    gt = np.zeros_like(pred)

    print('finished initializing matrices')

    print('starting to process data')
    for i, (x,y) in enumerate(tqdm.tqdm(iter(test_data.dataset),total=test_data.num_seqs)):
        chrom,s,e = seq_coords[i]
        target_interval = kipoiseq.Interval(chrom,s,e)

        x = one_hot_encode(fasta_extractor.extract(target_interval.resize(SEQUENCE_LENGTH)))

        pred_i = get_model_output(x,model)
        
        pred[i] = pred_i
        gt[i] = y.numpy()

    print('here')
    with open(f'model_best_test_set_gt.pickle','wb') as f:
        pickle.dump(gt,f)
    with open(f'model_best_test_set_pred.pickle','wb') as f:
        pickle.dump(pred,f)
        
################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
