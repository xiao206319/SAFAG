import torch
import numpy as np


def umeyama(X, Y):
    """
    Estimates the Sim(3) transformation between `X` and `Y` point sets.

    Estimates c, R and t such as c * R @ X + t ~ Y.

    Parameters
    ----------
    X : numpy.array
        (m, n) shaped numpy array. m is the dimension of the points,
        n is the number of points in the point set.
    Y : numpy.array
        (m, n) shaped numpy array. Indexes should be consistent with `X`.
        That is, Y[:, i] must be the point corresponding to X[:, i].

    Returns
    -------
    c : float
        Scale factor.
    R : numpy.array
        (3, 3) shaped rotation matrix.
    t : numpy.array
        (3, 1) shaped translation vector.
    """
    mu_x = X.mean(axis=1).reshape(-1, 1)
    mu_y = Y.mean(axis=1).reshape(-1, 1)
    var_x = np.square(X - mu_x).sum(axis=0).mean()
    cov_xy = ((Y - mu_y) @ (X - mu_x).T) / X.shape[1]
    U, D, VH = np.linalg.svd(cov_xy)
    S = np.eye(X.shape[0])
    if np.linalg.det(U) * np.linalg.det(VH) < 0:
        S[-1, -1] = -1
    c = np.trace(np.diag(D) @ S) / var_x
    R = U @ S @ VH
    t = mu_y - c * R @ mu_x
    return c.round(5), R.round(5), t.round(5)

def get_translation_per_batch(pts_list,pred_npcs_list):
    assert len(pts_list)==len(pred_npcs_list),'data loading failed!'
    batch_size = len(pts_list)
    pred_trans_list = torch.zeros((batch_size,3))
    for i in range(batch_size):
        pts = np.array(pts_list[i])
        npcs = np.array(pred_npcs_list[i])
        npcs = npcs - 0.5
        c,r,t = umeyama(pts.transpose(),npcs.transpose())
        t = torch.from_numpy(t).squeeze()
        pred_trans_list[i] = t
    return pred_trans_list

