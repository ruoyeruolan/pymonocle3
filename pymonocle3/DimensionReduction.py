"""
@Introduce  : 
@File       : DimensionReduction.py
@Author     : ryrl
@Email      : ryrl970311@gmail.com
@Time       : 2025/01/08 19:22
@Describe   : 
"""
import logging
import warnings
import scanpy as sc
from anndata import AnnData
from scipy.sparse import issparse
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD, IncrementalPCA,NMF


class DimensionReduction:

    def __init__(self,
                 model: str = 'pca',
                 center: bool = True,
                 scale: bool = True,
                 random_state: int = 42, **kwargs):
        """
        Initialize the dimension reduction model
        Parameters
        ----------
        model, optional
            model name, by default 'pca'
        center, optional
            whether or not to center the data, by default True
        scale, optional
            whether or not to scale the data, by default True
        random_state, optional, by default 42
        """
        
        super().__init__()
        self.scale = scale
        self.center = center
        self.kwargs = kwargs
        self.random_state = random_state

        self.model_name = model
        self.model = self._initialize_model(model)

    def _initialize_model(self, model: str = 'pca'):
        """
        Initialize the model for dimension reduction

        Parameters
        ----------
        model, str, optional
            model used to dimension reduction, reference to `sklearn.decomposition`, by default 'pca'

        Returns
        -------
            initialized model

        Raises
        ------
        ValueError
        """
        models = {
            'pca': PCA,
            'tsvd': TruncatedSVD,
            'ipca': IncrementalPCA,
            'nmf': NMF
        }

        if model not in models:
            raise ValueError(f'Invalid model: {model}')
        return models[model](random_state=self.random_state, **self.kwargs)
    
    def set_model(self, model: str, **kwargs):
        """
        update the model and its parameters

        Parameters
        ----------
        model: str
            model used to dimension reduction, reference to `sklearn.decomposition`
        """
        self.kwargs.update(kwargs)
        self.model = self._initialize_model(model)
    
    def center_scale(self, adata: AnnData):
        
        if issparse(adata.X) and self.center:
            warnings.warn(
                'The input data is sparse, centering will make it dense, '
                'if the data is too large, it may cause memory error.'
            )
        
        if self.scale:
            sc.pp.scale(adata, zero_center=self.center, **self.kwargs)

    def fit(self, adata: AnnData) -> AnnData:

        if not hasattr(self.model, 'fit'):
            raise AttributeError(f'{self.model} has no attribute fit')
        
        cls = self.model.fit(adata.X)
        adata.uns[self.model_name] = cls
        return adata
    
    def transform(self, adata: AnnData) -> AnnData:

        if not hasattr(self.model, 'transform'):
            raise AttributeError(f'{self.model} has no attribute transform')
        
        cls_ = self.model.transform(adata.X)
        adata.obsm[self.model_name] = cls_
        return adata