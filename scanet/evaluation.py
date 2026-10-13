"""Evaluation layer of the SCANet reproduction (Market-1501 protocol).
Behaviour follows the authors' image notebook (cells 12, 19, 21, 27 `test`) of
github.com/FatimaZulfiqar/Person-ReID-STCANet (commit 1b1eb9dd): feature = post-BN `f` (model output [1]),
no test-time flip, distance = 1 - cosine, Market-1501 metric (same-pid & same-camera gallery removed),
max_rank 50, k-reciprocal re-ranking (Zhong et al., CVPR 2017; k1=20, k2=6, lambda=0.3).
Plain and re-ranked results are always computed from the SAME features and reported separately.
Parity: tests/test_parity_eval.py. No model selection on the test set happens here."""
import numpy as np
import torch
import torch.nn.functional as F


def cosine_distance(a, b):
    return 1 - torch.mm(F.normalize(a, p=2, dim=1), F.normalize(b, p=2, dim=1).t())


@torch.no_grad()
def extract_features(model, loader, device="cuda"):
    """loader yields (imgs, pid, camid). Returns (features [N,D] float32 CPU, pids, camids)."""
    was_training = model.training
    model.eval()
    feats, pids, camids = [], [], []
    for imgs, pid, camid in loader:
        f = model(imgs.to(device, non_blocking=True))[1]
        feats.append(f.detach().float().cpu())
        pids.append(torch.as_tensor(pid))
        camids.append(torch.as_tensor(camid))
    model.train(was_training)
    return torch.cat(feats, 0), torch.cat(pids).numpy(), torch.cat(camids).numpy()


def eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    q_pids, g_pids = np.asarray(q_pids), np.asarray(g_pids)
    q_camids, g_camids = np.asarray(q_camids), np.asarray(g_camids)
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    all_cmc, all_AP, num_valid_q = [], [], 0.0
    for q_idx in range(num_q):
        q_pid, q_camid = q_pids[q_idx], q_camids[q_idx]
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            continue                                   # query identity absent from (filtered) gallery
        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.0
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.0) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        all_AP.append(tmp_cmc.sum() / num_rel)
    assert num_valid_q > 0, "all query identities are absent from the gallery"
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    return all_cmc, np.mean(all_AP)


def re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3):
    """k-reciprocal re-ranking (Zhong et al., CVPR 2017). Inputs: numpy distance matrices."""
    q_g_dist, q_q_dist, g_g_dist = (np.asarray(m) for m in (q_g_dist, q_q_dist, g_g_dist))
    original_dist = np.concatenate(
        [np.concatenate([q_q_dist, q_g_dist], axis=1),
         np.concatenate([q_g_dist.T, g_g_dist], axis=1)], axis=0)
    original_dist = np.power(original_dist, 2).astype(np.float32)
    original_dist = np.transpose(1.0 * original_dist / np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist).astype(np.float32)
    initial_rank = np.argsort(original_dist).astype(np.int32)
    query_num = q_g_dist.shape[0]
    gallery_num = q_g_dist.shape[0] + q_g_dist.shape[1]
    all_num = gallery_num
    for i in range(all_num):
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            cand_fwd = initial_rank[candidate, :int(np.around(k1 / 2.0)) + 1]
            cand_bwd = initial_rank[cand_fwd, :int(np.around(k1 / 2.0)) + 1]
            fi_candidate = np.where(cand_bwd == candidate)[0]
            candidate_k_reciprocal_index = cand_fwd[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2.0 / 3 * len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)
        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = 1.0 * weight / np.sum(weight)
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
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(
                V[i, indNonZero[j]], V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2.0 - temp_min)
    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    return final_dist[:query_num, query_num:]


def _metrics(distmat, q_pids, g_pids, q_camids, g_camids, ranks, max_rank):
    cmc, mAP = eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids, max_rank)
    out = {"mAP": float(mAP)}
    for r in ranks:
        out[f"R{r}"] = float(cmc[r - 1])
    out["cmc"] = cmc
    return out


def evaluate_features(qf, gf, q_pids, g_pids, q_camids, g_camids, ranks=(1, 5, 10, 20),
                      max_rank=50, rerank=True, k1=20, k2=6, lambda_value=0.3):
    """Returns {"plain": {...}} and, if rerank, {"rerank": {...}}, both from the same features."""
    distmat = cosine_distance(qf, gf).numpy()
    res = {"plain": _metrics(distmat, q_pids, g_pids, q_camids, g_camids, ranks, max_rank)}
    if rerank:
        qq = cosine_distance(qf, qf).numpy()
        gg = cosine_distance(gf, gf).numpy()
        dm = re_ranking(distmat, qq, gg, k1, k2, lambda_value)
        res["rerank"] = _metrics(dm, q_pids, g_pids, q_camids, g_camids, ranks, max_rank)
    return res
