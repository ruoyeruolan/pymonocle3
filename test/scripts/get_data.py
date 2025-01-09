# -*- encoding: utf-8 -*-
"""
@Introduce  :
@File       : get_data.py
@Author     : ryrl
@email      : ryrl970311@gmail.com
@Time       : 2025/1/7 14:48
@Describe   :
"""
import pandas as pd
import scanpy as sc
from pymonocle3.preprocess import preprocess_adata
from pymonocle3.utils import load_data, create_adata,perform_svd, estimate_size_factors, normalize_data


# Load data
adata = load_data(dirs='./test/data/pbmc4k')
adata = estimate_size_factors(adata, method='normalize')


# Create AnnData object
expression = pd.read_csv('./test/data/pbmc4k/filtered_gene_bc_matrices/hg19/matrix.mtx')



sc.pp.log1p(adata)

a = normalize_data(adata, method='log1p', copy=True)
normalize_data(adata, method='log1p', key_added='AAA')
sc.pp.scale()

adata_  = perform_svd(adata, n_components=50)

