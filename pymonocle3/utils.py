# -*- encoding: utf-8 -*-
"""
@Introduce  :
@File       : utils.py
@Author     : ryrl
@email      : ryrl970311@gmail.com
@Time       : 2025/1/6 16:58
@Describe   :
"""
import logging
import warnings

import inspect
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from pathlib import Path, PurePath
from typing import Literal, Optional
from scipy.sparse import csr_matrix, issparse
from sklearn.decomposition import TruncatedSVD, PCA

def load_data(dirs: str = None, fname: Path | str = None, **kwargs) -> AnnData:
    """
    Load single cell expression matrix from multi-types

    Parameters
    ----------
    dirs: Path to directory for `.mtx` and `.tsv` files, e.g. './filtered_gene_bc_matrices/hg19/'.
    fname: Path to the file, such as 10X_hd, h5ad

    Returns
    -------
    AnnData
    """

    if dirs and fname: raise ValueError('Cannot provide both `dirs` and `fname` at the same time.')

    if dirs: return sc.read_10x_mtx(path=dirs, **kwargs)

    if fname:
        suffix = PurePath(fname).suffix
        readers = {".h5": sc.read_10x_h5, ".h5ad": sc.read_h5ad,}

        if suffix in readers:
            return readers[suffix](filename=fname, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    raise ValueError("Either `dirs` or `fname` must be provided.")


def create_adata(
        expression_matrix: pd.DataFrame,
        cell_metadata: pd.DataFrame,
        gene_meta: pd.DataFrame,
        sparse: bool = True,
        **kwargs
) -> AnnData:
    """

    Parameters
    ----------
    expression_matrix: gene expression matrix
    cell_metadata:
    gene_meta: gene annotation
    sparse: `Ture` can save memory, default is True

    Returns
    -------
    A anndata object, n_cells * n_genes
    """
    assert tuple(expression_matrix.shape[0]) != (cell_metadata.shape[0], gene_meta.shape[0]), "Mismatch in shapes!"

    if sparse: expression_matrix = csr_matrix(expression_matrix)
    adata = AnnData(X=expression_matrix, obs=cell_metadata, var=gene_meta, **kwargs)
    return adata


def perform_svd(adata: AnnData,
                center: bool = True,
                scale: bool = True,
                n_components: int = 50,
                method: Literal['svd', 'pca'] = 'svd',
                algorithm: Literal['auto’, ‘full’, ‘covariance_eigh’, ''arpack', 'randomized'] = 'randomized', 
                **kwargs) -> AnnData:
    """
    Perform SVD on matrix
    Parameters
    ----------
    adata: AnnData, n_cells * n_genes
    center: bool, whether to center the data, default is True
            if `adata.X` is sparse, it will be converted to dense.
            if method is 'pca', center must be True.

    scale: bool, whether to scale the data, default is True
    n_components: int, number of components to keep, default is 50
    algorithm:  str, algorithm to use, default is 'randomized',
                can be 'auto’, 'full', 'covariance_eigh', 'arpack', 'randomized', default is 'randomized', 
                'arpack' or 'randomized' onnly for TruncatedSVD
    method: str, method to use, can be 'svd' or 'pca', default is 'svd'
    kwargs: additional arguments for `sklearn.decomposition.TruncatedSVD` and `scanpy.pp.scale`

    Returns
    -------
    AnnData, with `X_svd`, `right_`, `left_`, `svd` in `obsm`, `varm`, `uns`, respectively
    """
    
    if issparse(adata.X) and center:
       warnings.warn(
            'The input data is sparse, centering will make it dense, '
            'if the data is too large, it may cause memory error.'
        )
    
    if center is False and scale is True:
        logging.info('Just scaling the data...')

    if scale is True:
        sc.pp.scale(adata, zero_center=center, **kwargs)

    methods = {
        'svd': TruncatedSVD,
        'pca': PCA,
    }
    logging.info(f'Performing {method} with {n_components} components...')
    cls = methods[method](n_components=n_components, algorithm=algorithm, **kwargs)
    cls = cls.fit(adata.X)
    adata.uns[f'{method}_fit'] = cls
    return adata


def get_singular_values(adata: AnnData, method: Literal['svd', 'pca'] = 'svd'):

    if adata.uns.get(method) is None:
        raise ValueError(f'Please perform {method} first!')
    
    v = adata.uns[method].components_.T  # right singular vectors
    d = adata.uns[method].singluar_values_  # singular values
    
    if adata.obsm.get(method) is None:
        raise ValueError(f'Please perform transformmation by {method} first!')

    u = adata.obsm[method] / d[np.newaxis, :]  # left singular vectors
    return u, d, v


def normalize_data(adata: AnnData, method: Literal['log1p', 'size'] = 'log1p',
                   key_added: str | None = None, layer: str | None = None, **kwargs) -> Optional[AnnData]:
    """

    Parameters
    ----------
    adata:
        AnnData object, n_cells * n_genes
    method:
        method to normalize data, by default 'log1p'.
        `log1p`: log1p normalization
        `size`: size factor normalization
    key_added:
        Name of the field in `adata.obs` where the normalization factor is stored.
    layer:
        Layer to normalize instead of `X`. If `None`, `X` is normalized.
    kwargs:
        additional arguments for normalization, a dict, see `scanpy.pp.normalize_total`

    Returns
    -------
    Optional[AnnData]
        AnnData object if `copy=True` is passed in **kwargs.
        None if `copy=False` (modifies adata in place). 
    """

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
