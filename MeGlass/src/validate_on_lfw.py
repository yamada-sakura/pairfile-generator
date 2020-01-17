"""Validate a face recognizer on the "Labeled Faces in the Wild" dataset (http://vis-www.cs.umass.edu/lfw/).
Embeddings are calculated using the pairs from http://vis-www.cs.umass.edu/lfw/pairs.txt and the ROC curve
is calculated and plotted. Both the model metagraph and the model parameters need to exist
in the same directory, and the metagraph should have the extension '.meta'.
"""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import facenet
#import lfw
import kccs.lfw_kccs as lfw
import os
import sys
import math
import pathlib
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate

import matplotlib.pyplot as plt

def main(args):
    logs_dir = os.path.expanduser(args.logs_dir)
    if not os.path.isdir(logs_dir):
        os.makedirs(logs_dir)
    facenet.write_arguments_to_file(args, os.path.join(logs_dir, 'arguments.txt'))
    src_path,_ = os.path.split(os.path.realpath(__file__))
    facenet.store_revision_info(src_path, logs_dir, ' '.join(sys.argv)) # 'revision_info.txt'

    with tf.Graph().as_default():
        
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction, visible_device_list=args.visible_device_list)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            
            # Read the file containing the pairs used for testing
            # pairs = lfw.read_pairs(args.lfw_pairs)
            pairs_folder = "D:\\l4f\\dataset\\public\\cfp-dataset\\Protocol"
            face_dire = "/FP"
            pairs = lfw.make_pairs(pairs_folder, face_dire)

            # Get the paths for the corresponding images
            paths_all, actual_issame_all = lfw.get_paths(os.path.expanduser(args.cfpw_dir), pairs)

            #enamerate:(インデックス,値)を返す
            for pair_idx, (paths, actual_issame) in enumerate(zip(paths_all, actual_issame_all)):
                #print(pair_idx)
                #print(paths)
                #print(actual_issame)
                if args.emb_array is None or args.emb_array == 'None':
                    # Load the model
                    facenet.load_model(args.model, args.model_nrof_step)
                    
                    # 入力および出力テンソルを取得する
                    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                    
                    #image_size = images_placeholder.get_shape()[1]  # For some reason this doesn't work for frozen graphs
                    image_size = args.image_size
                    embedding_size = embeddings.get_shape()[1]
                
                    # フォワードパスを実行して埋め込み（顔ごとのベクトル）を計算する
                    print('Runnning forward pass on LFW images')
                    batch_size = args.lfw_batch_size
                    nrof_images = len(paths)
                    nrof_batches = int(math.ceil(1.0*nrof_images / batch_size))
                    emb_array = np.zeros((nrof_images, embedding_size))
                    for i in range(nrof_batches):
                        start_index = i*batch_size
                        end_index = min((i+1)*batch_size, nrof_images)
                        paths_batch = paths[start_index:end_index]
                        images = facenet.load_data(paths_batch, False, False, image_size)
                        feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                        emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
                else:
                    emb_array = np.load(args.emb_array)

                np.save('emb_array_' + str(pair_idx) + '.npy', emb_array)

                tpr, fpr, accuracy, th, val, val_std, far, th_far = lfw.evaluate(emb_array, 
                        actual_issame, nrof_folds=args.lfw_nrof_folds)

                print('Threshold: %1.3f+-%1.3f' % (np.mean(th), np.std(th)))
                print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
                print('Threshold FAR: %1.3f+-%1.3f' % (np.mean(th_far), np.std(th_far)))
                print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))

                auc = metrics.auc(fpr, tpr)
                print('Area Under Curve (AUC): %1.3f' % auc)	
                eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
                print('Equal Error Rate (EER): %1.3f' % eer)


                plt.figure()
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.xscale('log')
                plt.xlim(10**-5, 10**0)
                plt.subplot()
                label = 'AUC={auc:.6f}'.format(**{'auc': auc})
                plt.plot(fpr, tpr, label=label)
                plt.legend(loc='lower right').get_frame().set_alpha(0.8)
                plt.savefig('./roc_mean_' + str(pair_idx) + '.png')

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('cfpw_dir', type=str,
        help='Path to the data directory containing aligned LFW face patches.')
    parser.add_argument('--lfw_batch_size', type=int,
        help='Number of images to process in a batch in the LFW test set.', default=100)
    parser.add_argument('model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--lfw_file_ext', type=str,
        help='The file extension for the LFW dataset.', default='png', choices=['jpg', 'png'])
    parser.add_argument('--lfw_nrof_folds', type=int,
        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    parser.add_argument('--model_nrof_step', type=int,
        help='Number of step in model. (enable only model is a directory)', default=None)
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--visible_device_list', type=str,
        help='List of devices enabled, declare as comma-separated integer.', default="0")
    parser.add_argument('--emb_array', type=str,
        help='embedding npy file.', default=None)
    parser.add_argument('--logs_dir', type=str,
        help='log directory.', default='.')
    parser.add_argument('--all', action='store_false',
        help='do not split kccs data.')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
