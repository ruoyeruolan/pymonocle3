# -*- encoding: utf-8 -*-
"""
@Introduce:
@File: utils.py
@Author: ryrl
@email: ryrl970311@gmail.com
@Time: 2025/1/6 16:58
@Describe:
"""
import numpy as np
import pandas as pd
import scanpy as sc
from typing import Literal
from anndata import AnnData
from pathlib import Path, PurePath
from scipy.sparse import csr_matrix, issparse

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


def normalize_data(adata: AnnData, method: Literal['log1p']):
    pass

def estimate_size_factors(adata: AnnData, round_exprs: bool = True, method: Literal['log', 'normalize'] = 'log'):
    
    if round_exprs: 
        if issparse(adata.X):
            adata.X.data = np.round(adata.X.data).astype(int)
        else:
            adata.X = adata.X.round().astype(int)

    cell_sums = adata.X.sum(axis=1) # Note: sum of each cell

    if issparse(cell_sums):
        cell_sums = cell_sums.A.flatten()  # sparse matrix to dense array
    else:
        cell_sums = cell_sums.flatten()

    if (cell_sums == 0).any(): raise ValueError("Some cells have zero counts, cannot compute size factors.")

    # if method == 'normalize':
    #     sfs = cell_sums / np.exp(np.mean(np.log(cell_sums)))
    # elif method == 'log':
    #     sfs = np.log(cell_sums) / np.exp(np.mean(np.log(np.log(cell_sums))))
    # else:
    #     raise ValueError(f"Unsupported method: {method}")
    methods = {
        'normalize': cell_sums / np.exp(np.mean(np.log(cell_sums))),
        'log': np.log(cell_sums) / np.exp(np.mean(np.log(np.log(cell_sums)))),
    }

    if method not in methods: raise ValueError(f"Unsupported method: {method}")

    sfs = methods[method]
    sfs = np.nan_to_num(sfs, nan=1)
    return sfs