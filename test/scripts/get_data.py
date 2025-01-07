# -*- encoding: utf-8 -*-
"""
@Introduce  :
@File       : get_data.py
@Author     : ryrl
@email      : ryrl970311@gmail.com
@Time       : 2025/1/7 14:48
@Describe   :
"""
import scanpy as sc
from pymonocle3.utils import load_data, perform_svd
from pymonocle3.preprocess import estimate_size_factors, normalize_data

adata = load_data(dirs='./test/data/pbmc4k')
adata = estimate_size_factors(adata, method='normalize')

sc.pp.log1p(adata)

a = normalize_data(adata, method='log1p', copy=True)
normalize_data(adata, method='log1p', key_added='AAA')
sc.pp.scale()

adata_  = perform_svd(adata, n_components=50)