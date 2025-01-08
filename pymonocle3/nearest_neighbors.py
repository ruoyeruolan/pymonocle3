# -*- encoding: utf-8 -*-
"""
@Introduce  :
@File       : nearest_neighbors.py
@Author     : ryrl
@email      : ryrl970311@gmail.com
@Time       : 2025/1/8 21:41
@Describe   :
"""
import logging

import hnswlib
from anndata import AnnData
from annoy import AnnoyIndex
from scipy.sparse import issparse


def set_nn_control():
    return {
        'method': 'annoy',
        'metric': 'euclidean',
        'n_trees': 50,
        'k': 25,
        'M': 48,
        'ef_construction': 200,
        'ef': 150,
        'grain_size': 1,
        'cores': 1,
        'random_seed': 42,
    }

def make_nn_index(adata: AnnData, nn_control: dict | None = None, verbose: bool = False):  # TODO: Add more methods and parameters
    """
    Build a nearest neighbors index for the given data.
    Parameters
    ----------
    adata: AnnData, the data to build the index for
    nn_control: dict, parameters for the nearest neighbors index, reference to ``
    verbose

    Returns
    -------

    """
    if issparse(adata.X):
        data = adata.X.toarray()
    else:
        data = adata.X

    if nn_control is None:
        nn_control = set_nn_control()

    nrows, ncols = data.shape
    method = nn_control.get('method', 'annoy')
    metric = nn_control.get('metric', 'euclidean')
    n_trees = nn_control.get('n_trees', 10)
    # num_threads = nn_control.get('cores', 1)

    nn_control.update({
        'method': method,
        'metric': metric,
        'n_trees': n_trees,
        # 'num_threads': num_threads,
    })

    if verbose:
        print(f"Building {method} index with metric {metric}...")

    logging.info(f"Building {method} index ...")
    if method == 'annoy':
        # n_trees = nn_control.get('n_trees', 10)
        annoy_index = AnnoyIndex(ncols, metric)
        # annoy_index.set_seed(nn_control.get('random_seed', 42))
        for i in range(nrows):
            annoy_index.add_item(i, data[i])
        annoy_index.build(n_trees)
        return annoy_index, nn_control

    elif method == 'hnsw':
        M = nn_control.get('M', 16)
        ef_construction = nn_control.get('ef_construction', 200)
        hnsw_index = hnswlib.Index(space=metric, dim=ncols)
        hnsw_index.init_index(max_elements=nrows, ef_construction=ef_construction, M=M)
        hnsw_index.add_items(data, num_threads=-1)
        return hnsw_index, nn_control

    else:
        raise ValueError(f"Unsupported method: {method}")
