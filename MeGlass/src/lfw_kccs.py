import os
import lfw
import numpy as np
import pandas as pd

def evaluate(embeddings, actual_issame, nrof_folds=10):
    return lfw.evaluate(embeddings, actual_issame, nrof_folds)

def get_paths(dir, pairs_list):
    """
    paths = []
    issames = []
    for pairs in pairs_list:
        pair_shape = len(pairs[0])
        if pair_shape == 3 or pair_shape == 4:
            path_list, issame_list = lfw.get_paths(dir, pairs, file_ext) # lfw
            paths.append(path_list)
            issames.append(issame_list)
        elif pair_shape == 7:
            path_list, issame_list = get_paths_kccs(dir, pairs, file_ext) # kccs
            paths.append(path_list)
            issames.append(issame_list)
    return paths, issames
    """
    return lfw.get_paths(dir, pairs_list)

def get_paths_kccs(kccs_dir, pairs, file_ext):
    print('Kccs Data')
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    for pair in pairs:
        path0 = os.path.join(kccs_dir, pair[0], pair[1], pair[2] + '.' + file_ext)
        path1 = os.path.join(kccs_dir, pair[3], pair[4], pair[5] + '.' + file_ext)

        if pair[6] == "1":
            issame = True
        elif pair[6] == "0":
            issame = False

        if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist
            path_list += (path0,path1)
            issame_list.append(issame)
        else:
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs>0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)

    return path_list, issame_list

def make_pairs(pairs_filename, face_dire):
    """
    pairs = []
    for p in pairs_filename:
         pairs.append(lfw.read_pairs(os.path.expanduser(p)))
    return pairs
    """
    return lfw.make_pairs(pairs_filename, face_dire)