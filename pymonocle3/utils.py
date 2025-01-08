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
from typing import Literal
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
    v = cls.components_.T
    d = cls.singular_values_

    cls_ = cls.transform(adata.X)
    u = cls_ / d[np.newaxis, :]

    logging.info(f'Add SVD results to adata...')
    adata.obsm[f'X_{method}'] = cls_
    adata.varm['right_'] = v
    adata.obsm['left_'] = u
    adata.uns[method] = {    
        'algorithm': cls.algorithm,
        'n_components': cls.n_components,
        'n_features_in_': cls.n_features_in_,
        'singular_values': cls.singular_values_,
        'explained_variance_': cls.explained_variance_,
        'explained_variance_ratio_': cls.explained_variance_ratio_,
        
        'sdev': d / max(1, np.sqrt(adata.n_obs - 1)),  # NOTE: standard deviation, row is cell, col is gene
    } if method == 'svd' else {
        'mean_': cls.mean_,
        'n_samples_': cls.n_samples_,
        'n_features_': cls.n_features_,
        'singular_values_': cls.singular,
        'n_components_': cls.n_components_,
        'n_features_in_': cls.n_features_in_,
        'components_ndarray': cls.components_,
        'noise_variance_': cls.noise_variance_,
        'explained_variance_': cls.explained_variance_,
        'feature_names_in_': cls.get_feature_names_in_,
        'explained_variance_ratio_': cls.explained_variance_ratio_,
    } if method == 'pca' else None
    return adata

