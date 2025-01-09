# ****utf-8****
"""
@Introduce  : 
@File       : preprocess.py
@Author     : ryrl
@Email      : ryrl970311@gmail.com
@Time       : 2025/01/09 14:43
"""
import scanpy as sc
from pymonocle3.utils import load_data
from pymonocle3.preprocess import preprocess_adata

adata = load_data(dirs='./test/data/pbmc4k')
adata_ = preprocess_adata(adata, build_nn_index=True)

sc.pp.neighbors()
sc.pp.pca()