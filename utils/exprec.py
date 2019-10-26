import numpy as np
from sklearn.covariance import EmpiricalCovariance

from datasets import affectnet


def calc_mahalanobis_covs(annots):
    covs = []
    for expr_id, expr_name in enumerate(affectnet.AffectNet.classes):
        rows = annots.loc[annots['class'] == expr_id]
        X = rows.as_matrix(columns=['valence', 'arousal'])
        covs.append(EmpiricalCovariance().fit(X))
    return covs


def calc_centroids(annots):
    centroids = []
    for expr_id, expr_name in enumerate(affectnet.AffectNet.classes):
        rows = annots.loc[annots['class'] == expr_id]
        X = rows.as_matrix(columns=['valence', 'arousal'])
        centroids.append(np.mean(X, axis=0).astype(np.float32))
    return centroids


def get_expression_dists(samples, covs):
    dists = np.zeros((samples.shape[0], len(covs)))
    for cl,cov in enumerate(covs):
        dists[:, cl] = cov.mahalanobis(samples)
    return np.sqrt(dists)


def classify_expression(samples, covs):
    dists = get_expression_dists(samples, covs)
    return np.argmin(dists, axis=1)