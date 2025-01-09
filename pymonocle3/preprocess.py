# -*- encoding: utf-8 -*-
"""
@Introduce  : 
@File       : preprocess.py
@Author     : ryrl
@Email      : ryrl970311@gmail.com
@Time       : 2025/01/07 14:07
@Describe:
"""
import logging
import scanpy as sc
from anndata import AnnData
from typing import Optional

from pymonocle3.nearest_neighbors import make_nn_index
from pymonocle3.decomposition import DimensionReduction


def preprocess_adata(adata: AnnData, model: str = 'pca', n_components: int = 50,
        center: bool = True, scale: bool = True, 
        build_nn_index: bool = False, method: str = 'scanpy', nn_control: dict | None = None, verbose = False, 
        neighbor_kws: dict | None = None, **kwargs) -> AnnData:
    """
    Preprocess the AnnData.
    Parameters
    ----------
    adata
    model
    n_components
    center
    scale
    build_nn_index
    method
    nn_control
    verbose
    neighbor_kws
    kwargs

    Returns
    -------

    """
    if not isinstance(adata, AnnData):
        raise ValueError('adata must be an AnnData object.')

    logging.info(f"Normalizing data ...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)


    if build_nn_index:
        if method == 'monocle3':
            index, nn_control = make_nn_index(adata, nn_control, verbose)
            adata.uns['nn_index'] = index
            adata.uns['nn_control'] = nn_control
        
        elif method == 'scanpy':
            sc.pp.neighbors(adata, **neighbor_kws)  # TODO: Add parameters

        else:
            raise ValueError(f"Invalid method: {method}")
    # adata = adata[:, adata.X.sum(axis=0) > 0 & adata.X.sum(axis=1) != np.inf]
    # adata.layers['FM'] = adata.X.copy()[:, use_genes] if use_genes else adata.X.copy()

    # TODO: Add filter genes
    dim_reduction = DimensionReduction(model=model, n_components=n_components, center=center, scale=scale, **kwargs)
    adata = dim_reduction.fit(adata)
    adata = dim_reduction.transform(adata)
    return adata