# -*- encoding: utf-8 -*-
"""
@Introduce  : 
@File       : preprocess.py
@Author     : ryrl
@Email      : ryrl970311@gmail.com
@Time       : 2025/01/07 14:07
@Describe:
"""
import numpy as np
import scanpy as sc
from typing import Literal
from anndata import AnnData
from scipy.sparse import issparse


def normalize_data(adata: AnnData, method: Literal['log1p', 'size'],
                   key_added: str | None = None, layer: str | None = None, **kwargs):

    # copy = kwargs.get('copy', False)

    if method == 'log1p': return sc.pp.log1p(adata, layer=layer, **kwargs)

    elif method == 'size': return sc.pp.normalize_total(adata=adata, key_added=key_added, layer=layer, **kwargs)

    else: raise ValueError(f"Unsupported method: {method}")


def estimate_size_factors(adata: AnnData,
                          round_exprs: bool = False,
                          method: Literal['log', 'normalize'] = 'log') -> AnnData:
    """
    Estimate size factors for each cell.

    Parameters
    ----------
    adata: Anndata object, n_cells * n_genes
    round_exprs: whether to round expression values to integers, by default True
    method: method to compute size factors, by default 'log'

    Returns
    -------
    np.ndarray
        size factors for each cell

    Raises
    ------
    ValueError
        zero counts in some cells
    ValueError
        method is not supported
    """
    
    if round_exprs: 
        if issparse(adata.X):
            adata.X.data = np.round(adata.X.data).astype(int)
            adata.X.eliminate_zeros()
        else:
            adata.X = adata.X.round().astype(int)

    cell_sums = adata.X.sum(axis=1) # Note: sum of each cell

    if (cell_sums == 0).any(): raise ValueError("Some cells have zero counts, cannot compute size factors.")

    methods = {
        'normalize': cell_sums / np.exp(np.mean(np.log(cell_sums))),
        'log': np.log(cell_sums) / np.exp(np.mean(np.log(np.log(cell_sums)))),
    }

    if method not in methods: raise ValueError(f"Unsupported method: {method}")

    sfs = methods[method]
    sfs = np.nan_to_num(sfs, nan=1)
    adata.obs['SizeFactor'] = sfs
    return adata