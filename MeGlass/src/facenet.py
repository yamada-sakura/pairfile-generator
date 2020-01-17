"""Functions for building the face recognition network.
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

# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from subprocess import Popen, PIPE
import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
from scipy import misc
from sklearn.model_selection import KFold
from scipy import interpolate
from tensorflow.python.training import training
import random
import re
from tensorflow.python.platform import gfile
from six import iteritems
import pandas as pd
import cv2
from decimal import Decimal
from l4f.const_training import *
import common.mask_utils as mask_utils
import common.tf_utils as tf_utils

from sklearn import metrics
import matplotlib.pyplot as plt


def tf_random_normal_occlusion(image, occ_types=('plain', 'gray_gauss', 'rgb_gauss', 'speckle',
                                                 'nothing', 'nothing', 'nothing', 'nothing'),
                               occluder_height=20, occluder_width=40):
    """
    画像の一部に矩形マスクを施します。
    :param image: 入力画像
    :param occ_types: マスクの種類
    :param occluder_height: マスクの高さ
    :param occluder_width: マスクの幅
    :return: マスクを適用した画像
    """
    image_shape = tf.shape(image)
    image_height = image_shape[0]
    image_width = image_shape[1]
    begin_y, begin_x = tf_utils.pickup_point_normal(image_height - occluder_height, image_width - occluder_width)
    cells = tf_utils.grid_split(image, ys=[begin_y, occluder_height, -1], xs=[begin_x, occluder_width, -1])

    # create mask
    occ_index = tf.random_uniform((), 0, len(occ_types), dtype=tf.int64)
    # TODO: tf.contrib.lookup.index_to_string will be deprecated
    table = tf.contrib.lookup.index_to_string(
        occ_index, mapping=occ_types, default_value='nothing'
    )
    cells[1][1] = tf.case({
        tf.equal(table, tf.constant('nothing')): lambda: cells[1][1],
        tf.equal(table, tf.constant('gray')): lambda: tf_utils.gray_mask(cells[1][1]),
        tf.equal(table, tf.constant('plain')): lambda: tf_utils.plain_mask(cells[1][1]),
        tf.equal(table, tf.constant('gray_gauss')): lambda: tf_utils.gray_gauss_noise(cells[1][1]),
        tf.equal(table, tf.constant('rgb_gauss')): lambda: tf_utils.rgb_gauss_noise(cells[1][1]),
        tf.equal(table, tf.constant('speckle')): lambda: tf_utils.speckle_noise(cells[1][1])
        }, exclusive=True, default=lambda: cells[1][1], name='case_mask')

    masked = tf_utils.grid_concat(cells)
    return masked

def random_normal_occlusion(image, occ_type=None, occluder_height=20, occluder_width=40):
    """
    画像の一部に矩形マスクを施します。
    :param image: 入力画像
    :param occ_type: マスクの種類
    :param occluder_height: マスクの高さ
    :param occluder_width: マスクの幅
    :return: マスクを適用した画像
    """
    if isinstance(occ_type, bytes):
        occ_type = occ_type.decode('utf-8')
    occ_types = ['plain', 'gray_gauss', 'rgb_gauss', 'speckle']
    if occ_type is None or occ_type == '':
        occ_type = np.random.choice(occ_types)
    if occ_type not in occ_types:
        raise ValueError
    mask_utils.apply_occluder(image, occluder_height, occluder_width,
                                               occ_type=occ_type, method='normal')
    return image

def triplet_loss(anchor, positive, negative, alpha):
    """Calculate the triplet loss according to the FaceNet paper
    
    Args:
      anchor: the embeddings for the anchor images.
      positive: the embeddings for the positive images.
      negative: the embeddings for the negative images.
  
    Returns:
      the triplet loss according to the FaceNet paper as a float tensor.
    """
    with tf.variable_scope('triplet_loss'):
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)
        
        basic_loss = tf.add(tf.subtract(pos_dist,neg_dist), alpha)
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
      
    return loss
  
def decov_loss(xs):
    """Decov loss as described in https://arxiv.org/pdf/1511.06068.pdf
    'Reducing Overfitting In Deep Networks by Decorrelating Representation'
    """
    x = tf.reshape(xs, [int(xs.get_shape()[0]), -1])
    m = tf.reduce_mean(x, 0, True)
    z = tf.expand_dims(x-m, 2)
    corr = tf.reduce_mean(tf.matmul(z, tf.transpose(z, perm=[0,2,1])), 0)
    corr_frob_sqr = tf.reduce_sum(tf.square(corr))
    corr_diag_sqr = tf.reduce_sum(tf.square(tf.diag_part(corr)))
    loss = 0.5*(corr_frob_sqr - corr_diag_sqr)
    return loss 
  
def center_loss(features, label, alfa, nrof_classes):
    """Center loss based on the paper "A Discriminative Feature Learning Approach for Deep Face Recognition"
       (http://ydwen.github.io/papers/WenECCV16.pdf)
    """
    nrof_features = features.get_shape()[1]
    centers = tf.get_variable('centers', [nrof_classes, nrof_features], dtype=tf.float32,
        initializer=tf.constant_initializer(0), trainable=False)
    label = tf.reshape(label, [-1])
    centers_batch = tf.gather(centers, label)
    diff = (1 - alfa) * (centers_batch - features)
    centers = tf.scatter_sub(centers, label, diff)
    loss = tf.reduce_mean(tf.square(features - centers_batch))
    return loss, centers

def get_image_paths_and_labels(dataset):
    image_paths_flat = []
    labels_flat = []
    for i in range(len(dataset)):
        image_paths_flat += dataset[i].image_paths
        labels_flat += [i] * len(dataset[i].image_paths)
    return image_paths_flat, labels_flat

def shuffle_examples(image_paths, labels):
    shuffle_list = list(zip(image_paths, labels))
    random.shuffle(shuffle_list)
    image_paths_shuff, labels_shuff = zip(*shuffle_list)
    return image_paths_shuff, labels_shuff

def read_images_from_disk(input_queue):
    """Consumes a single filename and label as a ' '-delimited string.
    Args:
      filename_and_label_tensor: A scalar string tensor.
    Returns:
      Two tensors: the decoded image, and the string label.
    """
    label = input_queue[1]
    file_contents = tf.read_file(input_queue[0])
    example = tf.image.decode_image(file_contents, channels=3)
    return example, label

def get_black_eyemask(image, bb, lm, ellipse_major_r=20, thickness=40, prob_clipeye=0.5):
    """
    アイマスクを生成します。矩形のマスクまたは、矩形マスクから一定の確率で目の部分をくり抜いたマスクの2種類を生成します。
    :param image: 入力画像
    :param bb: 顔領域 (bounding box)
    :param lm: ランドマーク
    :param ellipse_major_r: くり抜きの水平方向の半径
    :param thickness: アイマスクの鉛直方向高さ
    :param prob_clipeye: 眼の部分をくり抜く確率
    :return: アイマスク
    """
    resizeMat = np.diag([image.shape[1],image.shape[0]] / (bb[2:4] - bb[0:2]))
    points = lm.reshape((5,2)).T
    points = points - bb[0:2].reshape((2,1))
    points_resized = np.matmul(resizeMat, points) # aligned face points

    eyevector = points_resized[:, CENTER_OF_LEFT_EYE] - points_resized[:, CENTER_OF_RIGHT_EYE]
    eyevector_ext_left = points_resized[:, CENTER_OF_LEFT_EYE] + eyevector * 5
    eyevector_ext_right = points_resized[:, CENTER_OF_RIGHT_EYE] - eyevector * 5
    theta = np.degrees(np.arctan(eyevector[1] / eyevector[0]))

    # init mask
    mask = np.zeros(image.shape, dtype=np.uint8)

    # add line around eye into mask
    points_resized_int = points_resized.astype(dtype=int)
    eyevector_ext_int = np.array([eyevector_ext_left, eyevector_ext_right]).astype(dtype=int)
    cv2.line(mask, tuple(eyevector_ext_int[0]), tuple(eyevector_ext_int[1]), (255,255,255), thickness)

    rand_clipeye = np.random.uniform(low=0.0, high=1.0)
    if rand_clipeye < prob_clipeye:
        ellipse_rc_ratio = 1.0 / 3.0
        major_r = ellipse_major_r
        minor_r = ellipse_major_r * ellipse_rc_ratio
        ellipse_rc = (int(major_r), int(minor_r))

        # add ellipse on eye
        cv2.ellipse(mask, tuple(points_resized_int[:, CENTER_OF_LEFT_EYE]), ellipse_rc, theta, 0, 360, (0,0,0), -1)
        cv2.ellipse(mask, tuple(points_resized_int[:, CENTER_OF_RIGHT_EYE]), ellipse_rc, theta, 0, 360, (0,0,0), -1)
    return mask

def random_gaussian_blind_image(image, bb, lm, ellipse_major_r=20, thickness=40, sigma=32):
    """
    顔画像に目隠しを施します
    :param image: 入力画像
    :param bb: 顔領域 (bounding box)
    :param lm: ランドマーク
    :param ellipse_major_r: くり抜きの水平方向の半径
    :param thickness: アイマスクの鉛直方向高さ
    :param sigma: アイマスク適用の確率分布(正規分布)の標準偏差
    :return: 低解像度画像または入力画像を出力
    """
    rand_apply = np.random.uniform(low=0.0, high=1.0)
    mask = get_black_eyemask(image, bb, lm, ellipse_major_r, thickness)
    if rand_apply < 0.5 and not (bb == np.empty(4)*np.nan).all() and not (lm == np.empty(10)*np.nan).all():
        rand_noise = np.random.normal(0, sigma, image.shape)
        noised_image = np.clip(image + rand_noise, 0, 255).astype(dtype=np.uint8) # clip to [0, 255]
        inter = cv2.bitwise_and(noised_image, mask)
        outer = cv2.bitwise_and(image, cv2.bitwise_not(mask))
        # apply mask
        image = cv2.bitwise_or(inter, outer)
    return image

def random_blind_image(image, bb, lm, ellipse_major_r=20, thickness=40):
    """
    顔画像に目隠しを施します
    :param image: 入力画像
    :param bb: 顔領域 (bounding box)
    :param lm: ランドマーク
    :param ellipse_major_r: くり抜きの水平方向の半径
    :param thickness: アイマスクの鉛直方向高さ
    :return: 低解像度画像または入力画像を出力
    """
    rand_apply = np.random.uniform(low=0.0, high=1.0)
    if rand_apply < 0.5 and not (bb == np.empty(4)*np.nan).all() and not (lm == np.empty(10)*np.nan).all():
        mask = get_black_eyemask(image, bb, lm, ellipse_major_r, thickness)
        # apply mask
        image = cv2.subtract(image, mask)
    return image

def random_reduce_resolution(image, prob_apply=0.5, scale_min=0.25, interpolation_downscale=cv2.INTER_AREA, interpolation_upscale=cv2.INTER_LINEAR):
    """
    元の画像から、解像度の低い画像を生成します
    画像の縮小率は離散一様分布により決定します
    :param image: 入力画像
    :param prob_apply: 低解像画像(p)と通常画像(1-p)の割合
    :param scale_min: 縮小率の最小値 (最大値は1(無変換))
    :param interpolation_downscale: 縮小アルゴリズム
    :param interpolation_upscale: 拡大アルゴリズム
    :return: 低解像度画像または入力画像を、p:1-pの割合で出力
    """
    rand_apply = np.random.uniform(low=0.0, high=1.0)
    if rand_apply < prob_apply:
        rand_scale = np.random.uniform(low=scale_min, high=1.0)
        shape_upsample = (image.shape[1], image.shape[0])
        shape_downsample = (int(image.shape[1] * rand_scale), int(image.shape[0] * rand_scale))
        downsampled = cv2.resize(image, shape_downsample, interpolation=interpolation_downscale)
        upsampled = cv2.resize(downsampled, shape_upsample, interpolation=interpolation_upscale)
        return upsampled
    return image

def drange(start, stop, step):
    """ range() の浮動小数点版
    """
    start = Decimal(str(start))
    step = Decimal(str(step))
    while(start < stop):
        yield(float(start))
        start = start + step

def random_reduce_resolution_discretely(image, prob_apply=0.5, scale_min=0.2, scale_max=0.6, scale_step=0.1, interpolation_downscale=cv2.INTER_AREA, interpolation_upscale=cv2.INTER_LINEAR):
    """
    元の画像から、解像度の低い画像を生成します
    画像の縮小率は離散一様分布により決定します
    :param image: 入力画像
    :param prob_apply: 低解像画像(p)と通常画像(1-p)の割合
    :param scale_min: 縮小率の最小値
    :param scale_max: 縮小率の最大値
    :param scale_step: 縮小率の間隔 (離散一様分布の間隔)
    :param interpolation_downscale: 縮小アルゴリズム
    :param interpolation_upscale: 拡大アルゴリズム
    :return: 低解像度画像または入力画像を、p:1-pの割合で出力
    """
    rand_apply = np.random.uniform(low=0.0, high=1.0)
    if rand_apply < prob_apply:
        scales = list(drange(scale_min, scale_max, scale_step))
        scale_index = np.random.randint(len(scales))  # uniform
        rand_scale = scales[scale_index]
        shape_upsample = (image.shape[1], image.shape[0])
        shape_downsample = (int(image.shape[1] * rand_scale), int(image.shape[0] * rand_scale))
        downsampled = cv2.resize(image, shape_downsample, interpolation=interpolation_downscale)
        upsampled = cv2.resize(downsampled, shape_upsample, interpolation=interpolation_upscale)
        return upsampled
    return image

def random_rotate_image(image):
    angle = np.random.uniform(low=-10.0, high=10.0)
    return misc.imrotate(image, angle, 'bicubic')
  
def draw_landmark(image, tbb, tlm):
    """
    ランドマークを描画します
    :param image: 入力画像
    :param tbb: 顔領域 (bounding box)
    :param tlm: ランドマーク
    :return: ランドマークが描画された画像
    """
    cv2.rectangle(image, tuple(tbb[0:2]), tuple(tbb[2:4]), (255, 0, 0), 3)
    cv2.circle(image, tuple(tlm[0:2]), 1, (0, 255, 255), 3)
    cv2.circle(image, tuple(tlm[2:4]), 1, (0, 255, 255), 3)
    cv2.circle(image, tuple(tlm[4:6]), 1, (255, 0, 255), 3)
    cv2.circle(image, tuple(tlm[6:8]), 1, (0, 255, 0), 3)
    cv2.circle(image, tuple(tlm[8:10]), 1, (0, 255, 0), 3)
    return image
  
def crop_based_on_landmark(image, tbb, tlm, cropped_size=160, padding_above_eye=10, vertical_center='center_of_image'):
    """
    ランドマークを基準として、画像をくり抜きます
    :param image: 入力画像
    :param tbb: 顔領域 (bounding box)
    :param tlm: ランドマーク
    :param cropped_size: 出力画像のサイズ
    :param padding_above_eye: 頭部(画像上方向)のパディング
    :param vertical_center: 画像の鉛直方向中心
    :return: くり抜き後の画像
    """
    offset_y = int(max(min(tlm[1] ,tlm[3]) - padding_above_eye, 0))
    # cut under the
    cropable_area_size = image.shape[0] - offset_y
    cropped = None
    try:
        if(not isinstance(vertical_center, str)):
            vertical_center = vertical_center.decode(sys.stdin.encoding)
        if(cropable_area_size >= cropped_size):
            if(vertical_center == 'center_of_image'):
                offset_x = int((image.shape[1] - cropped_size)//2)
            elif(vertical_center == 'nose'):
                offset_x = tlm[4] - cropped_size//2
                offset_x_min = 0
                offset_x_max = image.shape[1] - cropped_size
                offset_x = int(min(max(offset_x_min, offset_x), offset_x_max))
            elif(vertical_center == 'center_of_eyes'):
                offset_x = (tlm[0] + tlm[2] - cropped_size)//2
                offset_x_min = 0
                offset_x_max = image.shape[1] - cropped_size
                offset_x = int(min(max(offset_x_min, offset_x), offset_x_max))
            else:
                raise ValueError(vertical_center)
            cropped = image[offset_y:offset_y+cropped_size, offset_x:offset_x+cropped_size]
        else:
            if(vertical_center == 'center_of_image'):
                offset_x = int((image.shape[1] - cropable_area_size)//2)
            elif(vertical_center == 'nose'):
                offset_x = tlm[4] - cropable_area_size//2
                offset_x_min = 0
                offset_x_max = image.shape[1] - cropable_area_size
                offset_x = int(min(max(offset_x_min, offset_x), offset_x_max))
            elif(vertical_center == 'center_of_eyes'):
                offset_x = (tlm[0] + tlm[2] - cropable_area_size)//2
                offset_x_min = 0
                offset_x_max = image.shape[1] - cropable_area_size
                offset_x = int(min(max(offset_x_min, offset_x), offset_x_max))
            else:
                raise ValueError(vertical_center)
            cropped = image[offset_y:offset_y+cropable_area_size, offset_x:offset_x+cropable_area_size]
            cropped = cv2.resize(cropped, (cropped_size, cropped_size), interpolation=cv2.INTER_LINEAR)
    except:
        import traceback
        traceback.print_exc()
    return cropped
  
def read_and_augment_data(image_list, label_list, image_size, batch_size, max_nrof_epochs, 
        random_crop, random_flip, random_rotate, nrof_preprocess_threads, shuffle=True):
    
    images = ops.convert_to_tensor(image_list, dtype=tf.string)
    labels = ops.convert_to_tensor(label_list, dtype=tf.int32)
    
    # Makes an input queue
    input_queue = tf.train.slice_input_producer([images, labels],
        num_epochs=max_nrof_epochs, shuffle=shuffle)

    images_and_labels = []
    for _ in range(nrof_preprocess_threads):
        image, label = read_images_from_disk(input_queue)
        if random_rotate:
            image = tf.py_func(random_rotate_image, [image], tf.uint8)
        if random_crop:
            image = tf.random_crop(image, [image_size, image_size, 3])
        else:
            image = tf.image.resize_image_with_crop_or_pad(image, image_size, image_size)
        if random_flip:
            image = tf.image.random_flip_left_right(image)
        #pylint: disable=no-member
        image.set_shape((image_size, image_size, 3))
        image = tf.image.per_image_standardization(image)
        images_and_labels.append([image, label])

    image_batch, label_batch = tf.train.batch_join(
        images_and_labels, batch_size=batch_size,
        capacity=4 * nrof_preprocess_threads * batch_size,
        allow_smaller_final_batch=True)
  
    return image_batch, label_batch
  
def _add_loss_summaries(total_loss):
    """Add summaries for losses.
  
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
  
    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])
  
    # Attach a scalar summmary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name +' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))
  
    return loss_averages_op

def train(args, total_loss, global_step, optimizer, learning_rate, moving_average_decay, update_gradient_vars, log_histograms=True):
    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        if optimizer=='ADAGRAD':
            opt = tf.train.AdagradOptimizer(learning_rate)
        elif optimizer=='ADADELTA':
            opt = tf.train.AdadeltaOptimizer(learning_rate, rho=0.9, epsilon=1e-6)
        elif optimizer=='ADAM':
            opt = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
        elif optimizer=='RMSPROP':
            epsilon = args.epsilon if args.epsilon else 1.0
            opt = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=epsilon)
        elif optimizer=='MOM':
            opt = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
        else:
            raise ValueError('Invalid optimization algorithm')
    
        grads = opt.compute_gradients(total_loss, update_gradient_vars)
        
    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
  
    # Add histograms for trainable variables.
    if log_histograms:
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
   
    # Add histograms for gradients.
    if log_histograms:
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)
  
    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
  
    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')
  
    return train_op

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y  

def crop(image, random_crop, image_size):
    if image.shape[1]>image_size:
        sz1 = int(image.shape[1]//2)
        sz2 = int(image_size//2)
        if random_crop:
            diff = sz1-sz2
            (h, v) = (np.random.randint(-diff, diff+1), np.random.randint(-diff, diff+1))
        else:
            (h, v) = (0,0)
        image = image[(sz1-sz2+v):(sz1+sz2+v),(sz1-sz2+h):(sz1+sz2+h),:]
    return image
  
def flip(image, random_flip):
    if random_flip and np.random.choice([True, False]):
        image = np.fliplr(image)
    return image

def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret
  
def load_data(image_paths, do_random_crop, do_random_flip, image_size, do_prewhiten=True):
    """

    :param image_paths:path_batch
    :param do_random_crop: デフォルトFalse
    :param do_random_flip: デフォルトFalse
    :param image_size: デフォルト160
    :param do_prewhiten: デフォルトTrue
    :return:
    """
    nrof_samples = len(image_paths)
    images = np.zeros((nrof_samples, image_size, image_size, 3))
    for i in range(nrof_samples):
        img = misc.imread(image_paths[i])
        if img.ndim == 2:
            img = to_rgb(img)
        if do_prewhiten:
            img = prewhiten(img)
        img = crop(img, do_random_crop, image_size)
        img = flip(img, do_random_flip)
        images[i,:,:,:] = img
    return images

def get_label_batch(label_data, batch_size, batch_index):
    nrof_examples = np.size(label_data, 0)
    j = batch_index*batch_size % nrof_examples
    if j+batch_size<=nrof_examples:
        batch = label_data[j:j+batch_size]
    else:
        x1 = label_data[j:nrof_examples]
        x2 = label_data[0:nrof_examples-j]
        batch = np.vstack([x1,x2])
    batch_int = batch.astype(np.int64)
    return batch_int

def get_batch(image_data, batch_size, batch_index):
    nrof_examples = np.size(image_data, 0)
    j = batch_index*batch_size % nrof_examples
    if j+batch_size<=nrof_examples:
        batch = image_data[j:j+batch_size,:,:,:]
    else:
        x1 = image_data[j:nrof_examples,:,:,:]
        x2 = image_data[0:nrof_examples-j,:,:,:]
        batch = np.vstack([x1,x2])
    batch_float = batch.astype(np.float32)
    return batch_float

def get_triplet_batch(triplets, batch_index, batch_size):
    ax, px, nx = triplets
    a = get_batch(ax, int(batch_size/3), batch_index)
    p = get_batch(px, int(batch_size/3), batch_index)
    n = get_batch(nx, int(batch_size/3), batch_index)
    batch = np.vstack([a, p, n])
    return batch

def get_learning_rate_from_file(filename, epoch):
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.split('#', 1)[0]
            if line:
                par = line.strip().split(':')
                e = int(par[0])
                lr = float(par[1])
                if e <= epoch:
                    learning_rate = lr
                else:
                    return learning_rate
    raise ValueError('Schedule at epoch {} is undefined.'.format(epoch))

class ImageClass():
    "Stores the paths to images for a given class"
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths
  
    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'
  
    def __len__(self):
        return len(self.image_paths)


def transform_landmarks(bbs, lms, input_image_size):
    tbbs = np.zeros(bbs.shape, dtype=np.int32)
    tlms = np.zeros(lms.shape, dtype=np.int32)
    for i, (bb, lm) in enumerate(zip(bbs, lms)):
        src_bb_points_array = np.float32([[bb[0], bb[1]], [bb[2], bb[1]], [bb[0], bb[3]], [bb[2], bb[3]]])
        dst_bb = np.int32([0,0,input_image_size, input_image_size])
        dst_bb_points_array = np.float32([[dst_bb[0], dst_bb[1]], [dst_bb[2], dst_bb[1]], [dst_bb[0], dst_bb[3]], [dst_bb[2], dst_bb[3]]])
        # 4 points required.
        mat = cv2.getPerspectiveTransform(src_bb_points_array, dst_bb_points_array)

        src_lm_points_array = lm.reshape((5,2)).astype(np.float32)
        dst_lm_points_array = cv2.perspectiveTransform(np.array([src_lm_points_array]), mat).reshape((10))
        tbb = dst_bb.astype(np.int32)
        tlm = dst_lm_points_array.astype(np.int32)
        tbbs[i] = tbb
        tlms[i] = tlm
    return tbbs, tlms

def get_bb_and_lm(bbpaths, data_dirs, image_list, input_image_size=182):
    bb_df = None
    for data_dir, bbpath in zip(data_dirs, bbpaths):
        assert data_dir == os.path.dirname(bbpath), \
            'Bounding box file must put at {}, but you put at {}.'.format(os.path.join(data_dir, os.path.basename(bbpath)), bbpath)
        assert os.path.exists(bbpath), bbpath
        df = pd.read_csv(bbpath)
        if(bb_df is None):
            bb_df = df
        else:
            bb_df = pd.concat([bb_df, df], ignore_index=True)

    image_list_df = pd.DataFrame(data=image_list, columns=['ALIGHNED'])
    bb_df = pd.merge(image_list_df, bb_df, on='ALIGHNED', how='left')
    assert image_list_df['ALIGHNED'].tolist() == bb_df['ALIGHNED'].tolist(), 'Incorrect bounding box file is reffered.'
    bbs = bb_df[COLUMNS_BB].as_matrix()
    lms = bb_df[COLUMNS_LM].as_matrix()

    # TODO: Calculate tbbs, tlms before training, at aligning.
    tbbs = np.zeros(bbs.shape, dtype=np.int32)
    tlms = np.zeros(lms.shape, dtype=np.int32)
    for i, (bb, lm) in enumerate(zip(bbs, lms)):
        src_bb_points_array = np.float32([[bb[0], bb[1]], [bb[2], bb[1]], [bb[0], bb[3]], [bb[2], bb[3]]])
        dst_bb = np.int32([0,0,input_image_size, input_image_size])
        dst_bb_points_array = np.float32([[dst_bb[0], dst_bb[1]], [dst_bb[2], dst_bb[1]], [dst_bb[0], dst_bb[3]], [dst_bb[2], dst_bb[3]]])
        # 4 points required.
        mat = cv2.getPerspectiveTransform(src_bb_points_array, dst_bb_points_array)

        src_lm_points_array = lm.reshape((5,2)).astype(np.float32)
        dst_lm_points_array = cv2.perspectiveTransform(np.array([src_lm_points_array]), mat).reshape((10))
        tbb = dst_bb.astype(np.int32)
        tlm = dst_lm_points_array.astype(np.int32)
        tbbs[i] = tbb
        tlms[i] = tlm

    return bbs, lms, tbbs, tlms

def get_dataset_with_filter(paths, has_class_directories=True, include=None, exclude=None, image_regex=None):
    dataset = []

    if(not isinstance(paths, list)):
        paths = [paths]
    for path in paths:
        path_exp = os.path.expanduser(path)
        persondir_set = set(os.listdir(path_exp))
        # filter by list of person directories
        if(include and os.path.exists(include)):
            filter_set = set(pd.read_csv(include).ix[:,'person_name'])
            persondir_set = persondir_set.intersection(filter_set)
        if(exclude and os.path.exists(exclude)):
            filter_set = set(pd.read_csv(exclude).ix[:,'person_name'])
            persondir_set = persondir_set.symmetric_difference(filter_set)

        persondirs = list(persondir_set)
        classes = [re.search('id@[0-9a-z]+', persondir).group(0) for persondir in persondirs]
        classes = list(set(classes))

        # one class have MORE THAN ONE persondirs
        for class_name in classes:
            class_dirs = [persondir for persondir in persondirs if re.match(class_name, persondir)]

            image_paths = []
            for class_dir in class_dirs:
                paths = get_image_paths(os.path.join(path_exp, class_dir))
                if(image_regex):
                    cmp = re.compile(image_regex)
                    paths = [x for x in paths if cmp.search(x)]
                if len(paths) > 0:
                    image_paths += paths

            print("{} images is assinged in class {}.".format(len(image_paths), class_name))
            dataset.append(ImageClass(class_name, image_paths))
    return dataset

# FOR LFW DATASET
def get_dataset(path, has_class_directories=True):
    dataset = []
    path_exp = os.path.expanduser(path)
    classes = os.listdir(path_exp)
    classes.sort()
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        facedir = os.path.join(path_exp, class_name)
        image_paths = get_image_paths(facedir)
        dataset.append(ImageClass(class_name, image_paths))
  
    return dataset

def get_image_paths(persondir):
    image_paths = []
    if os.path.isdir(persondir):
        images = os.listdir(persondir)
        image_paths = [os.path.join(persondir, img) for img in images]
    return image_paths

def split_dataset(dataset, split_ratio, mode):
    if mode=='SPLIT_CLASSES':
        nrof_classes = len(dataset)
        class_indices = np.arange(nrof_classes)
        np.random.shuffle(class_indices)
        split = int(round(nrof_classes*split_ratio))
        train_set = [dataset[i] for i in class_indices[0:split]]
        test_set = [dataset[i] for i in class_indices[split:-1]]
    elif mode=='SPLIT_IMAGES':
        train_set = []
        test_set = []
        min_nrof_images = 2
        for cls in dataset:
            paths = cls.image_paths
            np.random.shuffle(paths)
            split = int(round(len(paths)*split_ratio))
            if split<min_nrof_images:
                continue  # Not enough images for test set. Skip class...
            train_set.append(ImageClass(cls.name, paths[0:split]))
            test_set.append(ImageClass(cls.name, paths[split:-1]))
    else:
        raise ValueError('Invalid train/test split mode "%s"' % mode)
    return train_set, test_set

def load_model(model, model_nrof_step=None):
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp, model_nrof_step)
        
        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)
      
        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file))
        saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))
    
def get_model_filenames(model_dir, model_nrof_step=None):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files)==0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files)>1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    meta_files = [s for s in files if '.ckpt' in s]

    print('model_nrof_step='+str(model_nrof_step))
    if(model_nrof_step is not None):
        compiler = re.compile(r'(^(model-[\w\- ]+.ckpt-{})\.)'.format(str(model_nrof_step)))
        target = list(filter(compiler.match, files))
        if len(target)==0:
            raise ValueError('Step {} was not found in the model directory ({})'.format(str(model_nrof_step), str(model_dir)))
        ckpt_file = compiler.search(target[0]).groups()[1]
    else:
        max_step = -1
        for f in files:
            step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
            if step_str is not None and len(step_str.groups())>=2:
                step = int(step_str.groups()[1])
                if step > max_step:
                    max_step = step
                    ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file

def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)
    
    tprs = np.zeros((nrof_folds,nrof_thresholds))
    fprs = np.zeros((nrof_folds,nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    th = np.zeros((nrof_folds))
    
    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff),1)
    indices = np.arange(nrof_pairs)
    
    plt.figure()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xscale('log')
    plt.xlim(10**-5, 10**0)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        
        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx], _, _, _, _ = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        th[fold_idx] = thresholds[best_threshold_index]
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx,threshold_idx], fprs[fold_idx,threshold_idx], _, _, _, _, _ = calculate_accuracy(threshold, dist[test_set], actual_issame[test_set])
        _, _, accuracy[fold_idx], _, _, _, _ = calculate_accuracy(thresholds[best_threshold_index], dist[test_set], actual_issame[test_set])

        plt.subplot()
        auc = metrics.auc(fprs[fold_idx], tprs[fold_idx])
        label = '{name:} AUC={auc:.6f}'.format(**{'name': str(fold_idx), 'auc': auc})
        plt.plot(fprs[fold_idx], tprs[fold_idx], label=label)

    plt.legend(loc='lower right').get_frame().set_alpha(0.8)
    i = 0
    while(True):
        if not os.path.exists('./roc_cross_' + str(i) + '.png'):
            plt.savefig('./roc_cross_' + str(i) + '.png')
            break
        else:
            i = i + 1


    ### Best Score
    tpr_all = np.zeros((nrof_thresholds))
    fpr_all = np.zeros((nrof_thresholds))
    acc_all = np.zeros((nrof_thresholds))
    for threshold_idx, threshold in enumerate(thresholds):
        tpr_all[threshold_idx], fpr_all[threshold_idx], acc_all[threshold_idx], _, _, _, _ = calculate_accuracy(threshold, dist, actual_issame)
    best_threshold_index = np.argmax(acc_all)
    best_th = thresholds[best_threshold_index]
    _, _, acc_best, tp, fp, tn, fn = calculate_accuracy(best_th, dist, actual_issame)
    print('■Best Score')
    print('正解率\t%1.3f' % acc_best)
    print('しきい値\t%1.3f' % best_th)
    print('TP, FN:\t%1.0f\t%1.0f' % (tp, fn))
    print('FP, TN:\t%1.0f\t%1.0f' % (fp, tn))
    print('')


    ### Smallest FPR
    tpr_fpr = np.zeros((nrof_thresholds))
    fpr_fpr = np.zeros((nrof_thresholds))
    acc_fpr = np.zeros((nrof_thresholds))
    acc_fpr = np.zeros((nrof_thresholds))
    for threshold_idx, threshold in enumerate(thresholds):
        _, fpr_fpr[threshold_idx], _, _, _, _, _ = calculate_accuracy(threshold, dist, actual_issame)
    smallest_fpr_threshold_index = [i for i, x in enumerate(fpr_fpr) if x == min(fpr_fpr)]
    for threshold_idx in smallest_fpr_threshold_index:
        smallest_fpr_th = thresholds[threshold_idx]
        _, _, acc_fpr[threshold_idx], _, _, _, _ = calculate_accuracy(smallest_fpr_th, dist, actual_issame)
    best_threshold_index = np.argmax(acc_fpr)
    best_th = thresholds[best_threshold_index]
    _, _, acc_fpr, tp_fpr, fp_fpr, tn_fpr, fn_fpr = calculate_accuracy(best_th, dist, actual_issame)
    print('■Smallest FPR Score')
    print('正解率\t%1.3f' % acc_fpr)
    print('しきい値\t%1.3f' % best_th)
    print('TP, FN:\t%1.0f\t%1.0f' % (tp_fpr, fn_fpr))
    print('FP, TN:\t%1.0f\t%1.0f' % (fp_fpr, tn_fpr))
    print('')


    ### 0.5 Score
    th_l4f = 0.5
    _, _, acc_l4f, tp_l4f, fp_l4f, tn_l4f, fn_l4f = calculate_accuracy(th_l4f, dist, actual_issame)
    print('■0.5 Score')
    print('正解率\t%1.3f' % acc_l4f)
    print('しきい値\t%1.3f' % 0.5)
    print('TP, FN:\t%1.0f\t%1.0f' % (tp_l4f, fn_l4f))
    print('FP, TN:\t%1.0f\t%1.0f' % (fp_l4f, tn_l4f))
    print('')


    tpr = np.mean(tprs,0)
    fpr = np.mean(fprs,0)
    return tpr, fpr, accuracy, th

def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    #print("====================================================================")
    #print(float(fp))
    #print("**********************************************************************")
    #print(float(fp+tn))
    tpr = float(tp) / float(tp+fn)
    fpr = float(fp) / float(fp+tn)
    acc = float(tp+tn)/dist.size
    return tpr, fpr, acc, tp, fp, tn, fn


  
def calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)
    
    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)
    th = np.zeros((nrof_folds))
    
    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff),1)
    indices = np.arange(nrof_pairs)
    
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
      
        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train)>=far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0
            print("far is under " + str(far_target))
        th[fold_idx] = threshold
        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist[test_set], actual_issame[test_set])
  

    ### Best Score
    print('■FAR Score')
    val_all = np.zeros((nrof_thresholds))
    far_all = np.zeros((nrof_thresholds))
    acc_all = np.zeros((nrof_thresholds))
    for threshold_idx, threshold in enumerate(thresholds):
        _, far_all[threshold_idx] = calculate_val_far(threshold, dist, actual_issame)
    if np.max(far_all)>=far_target:
        f = interpolate.interp1d(far_all, thresholds, kind='slinear')
        threshold = f(far_target)
    else:
        threshold = 0.0
        print("far is under " + str(far_target))
    _, _, acc, tp, fp, tn, fn = calculate_accuracy(threshold, dist, actual_issame)
    print('正解率\t%1.3f' % acc)
    print('しきい値\t%1.3f' % threshold)
    print('TP, FN:\t%1.0f\t%1.0f' % (tp, fn))
    print('FP, TN:\t%1.0f\t%1.0f' % (fp, tn))
    print('')


    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean, th


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far

def store_revision_info(src_path, output_dir, arg_string):
    try:
        # Get git hash
        cmd = ['git', 'rev-parse', 'HEAD']
        gitproc = Popen(cmd, stdout = PIPE, cwd=src_path)
        (stdout, _) = gitproc.communicate()
        git_hash = stdout.strip()
    except OSError as e:
        git_hash = ' '.join(cmd) + ': ' +  e.strerror
  
    try:
        # Get local changes
        cmd = ['git', 'diff', 'HEAD']
        gitproc = Popen(cmd, stdout = PIPE, cwd=src_path)
        (stdout, _) = gitproc.communicate()
        git_diff = stdout.strip()
    except OSError as e:
        git_diff = ' '.join(cmd) + ': ' +  e.strerror
    
    # Store a text file in the log directory
    rev_info_filename = os.path.join(output_dir, 'revision_info.txt')
    with open(rev_info_filename, "w") as text_file:
        text_file.write('arguments: %s\n--------------------\n' % arg_string)
        text_file.write('tensorflow version: %s\n--------------------\n' % tf.__version__)  # @UndefinedVariable
        text_file.write('git hash: %s\n--------------------\n' % git_hash)
        text_file.write('%s' % git_diff)

def list_variables(filename):
    reader = training.NewCheckpointReader(filename)
    variable_map = reader.get_variable_to_shape_map()
    names = sorted(variable_map.keys())
    return names

def put_images_on_grid(images, shape=(16,8)):
    nrof_images = images.shape[0]
    img_size = images.shape[1]
    bw = 3
    img = np.zeros((shape[1]*(img_size+bw)+bw, shape[0]*(img_size+bw)+bw, 3), np.float32)
    for i in range(shape[1]):
        x_start = i*(img_size+bw)+bw
        for j in range(shape[0]):
            img_index = i*shape[0]+j
            if img_index>=nrof_images:
                break
            y_start = j*(img_size+bw)+bw
            img[x_start:x_start+img_size, y_start:y_start+img_size, :] = images[img_index, :, :, :]
        if img_index>=nrof_images:
            break
    return img

def write_arguments_to_file(args, filename):
    with open(filename, 'w') as f:
        for key, value in iteritems(vars(args)):
            f.write('%s: %s\n' % (key, str(value)))
