import pyper
import math
import numpy as np


# Wrapper class of copula implemented in R.
class Copula:
    # if 'param' is None, the parameter of copula is estimated in R.
    def __init__(self, training_marginal_dist_matrix: np.matrix, family: str, param=None, dim=None):
        if family not in ['gumbel', 'clayton', 'frank', 'normal', 'indep']:
            print('Copula family "' + family + '" is not supported.')
            raise ValueError
        if training_marginal_dist_matrix.size == 0:
            if dim is None:
                raise ValueError
            self._dimension = dim
        else:
            self._dimension = training_marginal_dist_matrix.shape[1]
        self._r_engine = pyper.R()
        self._r_engine.assign("py.training.marginal.dist.matrix", training_marginal_dist_matrix)
        self._r_engine.assign("py.cop.name", family)
        self._r_engine.assign("py.param", param)
        self._r_engine.assign("py.dim", self._dimension)
        self._r_engine('source("copula/copula.R")')
        self._r_engine('source("copula/copula.R")')
        trained_param = self._r_engine.get("trained.param")
        self.trained_param = trained_param#asahi
        if trained_param is None:
            self._r_engine('trained <- indepCopula(dim=%d)' % self._dimension)
            print('indep')
        else:
            print(trained_param)

    # Caution! This method is not thread safe.
    def cdf(self, marginal_dist_matrix: np.matrix) -> float:
        if self._dimension != marginal_dist_matrix.shape[1]:
            return None
        self._r_engine.assign("py.marginal.dist.matrix", marginal_dist_matrix)
        cdf = self._r_engine.get("pCopula(py.marginal.dist.matrix, trained)")
        if cdf is None:
            return 0
        if math.isnan(cdf):
            return 0
        return cdf

    # Caution! This method is not thread safe.
    def pdf(self, marginal_dist_matrix: np.matrix) -> float:
        if self._dimension != marginal_dist_matrix.shape[1]:
            return None
        self._r_engine.assign("py.marginal.dist.matrix", marginal_dist_matrix)
        pdf = self._r_engine.get("dCopula(py.marginal.dist.matrix, trained)")
        if pdf is None:
            return 0
        if math.isnan(pdf):
            return 0
        return pdf

    def get_optimized_param(self) -> float:
        return self._r_engine.get("trained.param")

    def get_param(self)->dict:
        return {'trained_param':self.trained_param,'dimension':self._dimension}
