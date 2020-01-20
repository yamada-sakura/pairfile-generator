"""Helper for evaluation on the Labeled Faces in the Wild dataset 
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

import os
import numpy as np
import facenet
import glob

def evaluate(embeddings, actual_issame, nrof_folds=10):
    """
       評価指標をそれぞれ計算します
       :param embeddings:ゼロnumpy配列のラベルバッチの場所にテンソルを入れたもの
       :param actual_issame:ペアが他人かどうかのラベル（T/F配列）
       :param nrof_folds:交差検証でしようする折り目の数（デフォ：10）
       :param distance_metric:距離測定（デフォ：ユークリッド距離）
       :param subtract_mean: 距離を計算する前に、特徴の平均値を減算するかどうか（デフォ：する）
       :return:tpr（テストの時の予想とラベルが正解した割合の平均）
               fpr（テストの時の予想とラベルが不正解した割合の平均）
               accuracy（テストの時の予想とラベルが正解した割合の平均）
               val（テストデータにおいてラベルがTrueのうち予想がTrueで正解できた割合の平均）
               val_std（varの標準僅差）
               far（テストデータにおいてラベルがFalseのうち予想がTrueで不正解になった割合の平均）
    """
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy, th = facenet.calculate_roc(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), nrof_folds=nrof_folds)
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far, th_far = facenet.calculate_val(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds)
    return tpr, fpr, accuracy, th, val, val_std, far, th_far

def get_paths(cfpw_dir, pairs):
    """
       画像データまでのパスを作成し、リスト化します
       :param cfpw_dir:整列したデータディレクトリへのパス作成したペアリスト(D:\l4f\dataset\public\cfp-dataset\Data\Images)
       :param pairs:例['../Data/Images/488/frontal/04.jpg', '../Data/Images/488/profile/04.jpg', 1]
       :return:path_list(ペア内の画像データまでのパスをひたすらくっつけたリスト)と
               issame_list(ペアファイルが他人かどうかよってtrue/falseをくっつけたリスト)を返します
    """
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    #cfpw_dir = cfpw_dir.replace('/',os.sep)
    for pair in pairs:
        #print("************pair******************")
        #print(pair)
        path0 = os.path.join(cfpw_dir, os.path.relpath(pair[0],start="../Data/Images/"))
        path1 = os.path.join(cfpw_dir, os.path.relpath(pair[1],start="../Data/Images/"))
        #print("**********path0,path1**************")
        #print(path0)
        #print(path1)
        if pair[2] == 0:
            issame = False
        elif pair[2] == 1:
            issame = True
        # exists:ファイルの存在確認
        if os.path.exists(path0) and os.path.exists(path1):
            path_list += (path0, path1)
            issame_list.append(issame)
        else:
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs > 0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)

    #print(path_list)
    #print(len(path_list))
    #print(issame_list)
    #print(len(issame_list))
    return [path_list], [issame_list]

def read_pairs(pairs_filename):
    """
    ペアファイルからペアを要素としてリストを作る
    :param pairs_filename: ペアファイル.txt
    :return: ペアを要素としてしきつめたリスト
    """
    count = 0
    pairs = []
    twoline = []
    with open("dataset/calfw/pairs_CALFW.txt", 'r') as f:
        for line in f.readlines()[:]:
            count += 1
            text = line.rstrip()
            split_list = text.split()
            if count % 2 != 0:
                twoline.append(split_list[0])
                twoline.append(split_list[1])
            elif count % 2 == 0:
                twoline.append(split_list[0])
                twoline.append(split_list[1])
                pairs.append(twoline)
                twoline = []
            else:
                f.close()
    # print(pairs)
    return np.array(pairs)

def make_pairs(pairs_folder,face_dire):
    """
    ペアファイル構成フォルダからペアリストを作成する
    :param pairs_folder: ペアファイル構成フォルダ
    :param face_dire：　顔向きの組み合わせ（FF,FP）
    :return:FPペアリスト
    """
    pairs_folder = os.path.normpath(pairs_folder)

    # Pair_list_F.txt、Pair_list_P.txtをリスト化する
    # 1 ../Data/Images/001/frontal/01.jpg、　2 ../Data/Images/001/frontal/02.jpg、、、
    f_list = []
    p_list = []
    with open(pairs_folder + "/png" + "/[png]Pair_list_F.txt",'r') as f:
        for line in f.readlines()[:]:
            # text_list：[1, ../Data/Images/001/frontal/01.jpg]
            text_list = line.strip().split()
            f_list.append(text_list)

    with open(pairs_folder + "/png" + "/[png]Pair_list_P.txt",'r') as f:
        for line in f.readlines()[:]:
            # text_list：[1, ../Data/Images/001/frontal/01.jpg]
            text_list = line.strip().split()
            p_list.append(text_list)

    #print("******************Fリスト表示********************")
    #print(f_list)
    #print("******************Pリスト表示********************")
    #print(p_list)

    # 不一致、一致、不一致、、、という風にペアリストを作成する
    # 1行分(1要素)の構成例[../Data/Images/001/frontal/01.jpg, ../Data/Images/001/frontal/02.jpg, 0]
    pair = []
    pairs = []
    # 01,02,03,,,,10
    for set_name in sorted(glob.glob(pairs_folder + "/Split" + face_dire + "/*")):
        print(set_name)
        # diff.txt⇒0、same.txt⇒1
        with open(set_name + "/diff.txt", 'r') as f:
            # 5,97\n3,94\n、、、全350行
            for line in f.readlines()[:]:
                ds_num = 0
                line_list = line.strip().split(",")
                pair.append(f_list[int(line_list[0]) - 1][1])
                pair.append(p_list[int(line_list[1]) - 1][1])
                pair.append(ds_num)
                pairs.append(pair)
                pair = []
        with open(set_name + "/same.txt", 'r') as f:
            # 5,97\n 3,94\n 、、、全350行
            for line in f.readlines()[:]:
                ds_num = 1
                line_list = line.strip().split(",")
                pair.append(f_list[int(line_list[0]) - 1][1])
                pair.append(p_list[int(line_list[1]) - 1][1])
                pair.append(ds_num)
                pairs.append(pair)
                pair = []

    #print("****************print(pairs)表示*************************")
    #print(pairs)

    return pairs






