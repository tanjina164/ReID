"""Evaluation for SCAN-ReID.
cosine_distance / compute_distance_matrix / re_ranking / eval_market1501 / evaluate_model are copied
VERBATIM from the authors' image notebook (FatimaZulfiqar/Person-ReID-STCANet, commit 1b1eb9dd),
cells 29, 45, 49. Only change: `from __future__` lines removed (no-op in Python 3).
compute_minp (bottom of file) is NOT in the authors' code; it is added by the reproducer.
"""

import torch
from torch.nn import functional as F


def compute_distance_matrix(input1, input2, metric): # default metric = 'euclidean'
    
    """A wrapper function for computing distance matrix.
    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.
        metric (str, optional): "euclidean" or "cosine".
            Default is "euclidean".
    Returns:
        torch.Tensor: distance matrix.
    Examples::
       >>> from torchreid import metrics
       >>> input1 = torch.rand(10, 2048)
       >>> input2 = torch.rand(100, 2048)
       >>> distmat = metrics.compute_distance_matrix(input1, input2)
       >>> distmat.size() # (10, 100)
    """
    # check input
    assert isinstance(input1, torch.Tensor)
    assert isinstance(input2, torch.Tensor)
    assert input1.dim() == 2, 'Expected 2-D tensor, but got {}-D'.format(
        input1.dim()
    )
    assert input2.dim() == 2, 'Expected 2-D tensor, but got {}-D'.format(
        input2.dim()
    )
    assert input1.size(1) == input2.size(1)

    if metric == 'euclidean':
        distmat = euclidean_squared_distance(input1, input2)
    elif metric == 'cosine':
        distmat = cosine_distance(input1, input2)
    else:
        raise ValueError(
            'Unknown distance metric: {}. '
            'Please choose either "euclidean" or "cosine"'.format(metric)
        )

    return distmat


def euclidean_squared_distance(input1, input2):
    """Computes euclidean squared distance.
    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.
    Returns:
        torch.Tensor: distance matrix.
    """
    m, n = input1.size(0), input2.size(0)
    mat1 = torch.pow(input1, 2).sum(dim=1, keepdim=True).expand(m, n)
    mat2 = torch.pow(input2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat = mat1 + mat2
    distmat.addmm_(input1, input2.t(), beta=1, alpha=-2)
    return distmat


def cosine_distance(input1, input2):
    """Computes cosine distance.
    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.
    Returns:
        torch.Tensor: distance matrix.
    """
    input1_normed = F.normalize(input1, p=2, dim=1)
    input2_normed = F.normalize(input2, p=2, dim=1)
    distmat = 1 - torch.mm(input1_normed, input2_normed.t())
    return distmat

"""
q_g_dist: query-gallery distance matrix, numpy array, shape [num_query, num_gallery]
q_q_dist: query-query distance matrix, numpy array, shape [num_query, num_query]
g_g_dist: gallery-gallery distance matrix, numpy array, shape [num_gallery, num_gallery]
k1, k2, lambda_value: parameters, the original paper is (k1=20, k2=6, lambda_value=0.3)
Returns:
  final_dist: re-ranked distance, numpy array, shape [num_query, num_gallery]
"""

import numpy as np

__all__ = ['re_ranking']


def re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3):

    # The following naming, e.g. gallery_num, is different from outer scope.
    # Don't care about it.
    original_dist = np.concatenate(
        [
            np.concatenate([q_q_dist, q_g_dist], axis=1),
            np.concatenate([q_g_dist.T, g_g_dist], axis=1)
        ],
        axis=0
    )
    original_dist = np.power(original_dist, 2).astype(np.float32)
    original_dist = np.transpose(
        1. * original_dist / np.max(original_dist, axis=0)
    )
    V = np.zeros_like(original_dist).astype(np.float32)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    query_num = q_g_dist.shape[0]
    gallery_num = q_g_dist.shape[0] + q_g_dist.shape[1]
    all_num = gallery_num

    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[
                candidate, :int(np.around(k1 / 2.)) + 1]
            candidate_backward_k_neigh_index = initial_rank[
                candidate_forward_k_neigh_index, :int(np.around(k1 / 2.)) + 1]
            fi_candidate = np.where(
                candidate_backward_k_neigh_index == candidate
            )[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[
                fi_candidate]
            if len(
                np.
                intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)
            ) > 2. / 3 * len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(
                    k_reciprocal_expansion_index, candidate_k_reciprocal_index
                )

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = 1. * weight / np.sum(weight)
    original_dist = original_dist[:query_num, ]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float32)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float32)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1, gallery_num], dtype=np.float32)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(
                V[i, indNonZero[j]], V[indImages[j], indNonZero[j]]
            )
        jaccard_dist[i] = 1 - temp_min / (2.-temp_min)

    final_dist = jaccard_dist * (1-lambda_value) + original_dist*lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num, query_num:]
    
    return final_dist


import numpy as np
import copy
from collections import defaultdict
import sys

try:
    from eval_lib.cython_eval import eval_market1501_wrap
    CYTHON_EVAL_AVAI = True
    print("Cython evaluation is AVAILABLE")
except ImportError:
    CYTHON_EVAL_AVAI = False
    print("Warning: Cython evaluation is UNAVAILABLE")


def eval_cuhk03(distmat, q_pids, g_pids, q_camids, g_camids, max_rank, N=100):
    """Evaluation with cuhk03 metric
    Key: one image for each gallery identity is randomly sampled for each query identity.
    Random sampling is performed N times (default: N=100).
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0. # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        orig_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        kept_g_pids = g_pids[order][keep]
        g_pids_dict = defaultdict(list)
        for idx, pid in enumerate(kept_g_pids):
            g_pids_dict[pid].append(idx)

        cmc, AP = 0., 0.
        for repeat_idx in range(N):
            mask = np.zeros(len(orig_cmc), dtype=np.bool)
            for _, idxs in g_pids_dict.items():
                # randomly sample one image for each gallery person
                rnd_idx = np.random.choice(idxs)
                mask[rnd_idx] = True
            masked_orig_cmc = orig_cmc[mask]
            _cmc = masked_orig_cmc.cumsum()
            _cmc[_cmc > 1] = 1
            cmc += _cmc[:max_rank].astype(np.float32)
            # compute AP
            num_rel = masked_orig_cmc.sum()
            tmp_cmc = masked_orig_cmc.cumsum()
            tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
            tmp_cmc = np.asarray(tmp_cmc) * masked_orig_cmc
            AP += tmp_cmc.sum() / num_rel
        cmc /= N
        AP /= N
        all_cmc.append(cmc)
        all_AP.append(AP)
        num_valid_q += 1.

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


def eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids, max_rank):
    """Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0. # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        orig_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


def evaluate_model(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50, use_metrics_cuhk03=False, use_cython=True):
    if use_metrics_cuhk03:
        return eval_cuhk03(distmat, q_pids, g_pids, q_camids, g_camids, max_rank)
    else:
        if use_cython and CYTHON_EVAL_AVAI:
            return eval_market1501_wrap(distmat, q_pids, g_pids, q_camids, g_camids, max_rank)
        else:
            return eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids, max_rank)

# ---- Added by the reproducer (NOT authors' code) ----
def compute_minp(distmat, q_pids, g_pids, q_camids, g_camids):
    """mINP: mean over valid queries of (#positives / rank of the LAST positive).
    Same gallery filtering as eval_market1501 (same pid AND same camera removed)."""
    num_q, num_g = distmat.shape
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    all_inp = []
    for q_idx in range(num_q):
        order = indices[q_idx]
        remove = (g_pids[order] == q_pids[q_idx]) & (g_camids[order] == q_camids[q_idx])
        orig_cmc = matches[q_idx][np.invert(remove)]
        if not np.any(orig_cmc):
            continue
        pos = np.where(orig_cmc == 1)[0]
        all_inp.append(len(pos) / (pos[-1] + 1.0))
    return float(np.mean(all_inp))
