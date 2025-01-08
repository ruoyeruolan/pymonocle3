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
import numpy as np
from annoy import AnnoyIndex
from scipy.sparse import issparse, csr_matrix

def make_nn_index(data: np.array | csr_matrix, nn_control: dict, verbose: bool = False):
    """
    Build a nearest neighbors index for the given data.
    Parameters
    ----------
    data: np.ndarray or csr_matrix, data to build the index
    nn_control: dict, parameters for the nearest neighbors index, reference to ``
    verbose

    Returns
    -------

    """
    # data = adata.X.toarray() if issparse(adata.X) else adata.X
    if issparse(data):
        data = data.toarray()

    nrows, ncols = data.shape
    method = nn_control.get('method', 'annoy')
    metric = nn_control.get('metric', 'euclidean')

    if verbose:
        print(f"Building {method} index with metric {metric}...")
    
    logging.info(f"Building {method} index ...")
    if method == 'annoy':
        n_trees = nn_control.get('n_trees', 10)
        annoy_index = AnnoyIndex(ncols, metric)
        for i in range(nrows):
            annoy_index.add_item(i, data[i])
        annoy_index.build(n_trees)
        return annoy_index
    
    elif method == 'hnsw':
        M = nn_control.get('M', 16)
        ef_construction = nn_control.get('ef_construction', 200)
        hnsw_index = hnswlib.Index(space=metric, dim=ncols)
        hnsw_index.init_index(max_elements=nrows, ef_construction=ef_construction, M=M)
        hnsw_index.add_items(data)
        return hnsw_index

    else:
        raise ValueError(f"Unsupported method: {method}")

