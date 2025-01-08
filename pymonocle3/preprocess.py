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

from nearest_neighbors import make_nn_index
from decomposition import DimensionReduction


def preprocess_adata(adata: AnnData, model: str = 'pca', n_components: int = 50,
        center: bool = True, scale: bool = True, 
        build_nn_index: bool = False, nn_control: dict | None = None, verbose = False,
        use_genes: Optional[list[str]] = None, **kwargs) -> AnnData:
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
    nn_control
    verbose
    use_genes
    kwargs

    Returns
    -------

    """
    if not isinstance(adata, AnnData):
        raise ValueError('adata must be an AnnData object.')
    
    if build_nn_index and nn_control is not None:
        index = make_nn_index(adata, nn_control, verbose)
        adata.uns['nn_index'] = index
        adata.uns['nn_control'] = nn_control
    else:
        raise ValueError('Cannot build nearest neighbors index without nn_control.')
    
    logging.info(f"Normalizing data ...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # adata = adata[:, adata.X.sum(axis=0) > 0 & adata.X.sum(axis=1) != np.inf]
    adata.layers['FM'] = adata.X.copy()[:, use_genes] if use_genes else adata.X.copy()

    dim_reduction = DimensionReduction(model=model, n_components=n_components, center=center, scale=scale, **kwargs)
    adata = dim_reduction.fit(adata)
    adata = dim_reduction.transform(adata)
    return adata