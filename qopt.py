
import numpy as np
import pandas as pd
from cvxopt import matrix, solvers
from .base import BaseOptimizer

solvers.options['abstol'] = 1e-8
solvers.options['reltol'] = 1e-7


class QOptimizer(BaseOptimizer):
    """
    Mathematically, QOptimizer tries to solve the following problem:

    Maximize: a^T * h - λ_cov * (h - h_bmk)^T * V * (h - h_bmk)
            - λ_norm * ( α||h - h_0||_1 + (1 - α)||h - h_0||_2) - OtherTerms
    Subject to: lower <= h <= upper,
                sum(h) = 1,
                OtherConstraints

    where h is the target weight vector of length of the universe and
          h_bmk is the weight vector of the benchmark (e.g. index)
    """

    def __init__(self, universe, **kwargs):
        self.u = pd.Index(universe).drop_duplicates().sort_values()
        self.u_size = len(self.u)
        self.lower_limit = np.zeros(self.u_size)
        self.upper_limit = np.ones(self.u_size)
        self.total_limit = 1.0

        if (benchmark := kwargs.pop('benchmark', None)) is not None:
            self.set_benchmark(benchmark)

        return

    def set_limit(self, label, limit, overwrite=None):
        if isinstance(limit, int) or isinstance(limit, float):
            if label == 'total':
                self.total_limit = limit
                return

            target_limit = np.full(self.u_size, limit)
        elif isinstance(limit, dict) or isinstance(limit, pd.Series):
            target_limit = pd.Series(limit).reindex(
                self.u, fill_value=0.0)
        else:
            assert len(limit) == len(self.u), \
                f"{label}_limit should be a scalar or an array of size of the universe"
            target_limit = np.array(limit)

        if overwrite is not None:
            overwrite_limits = pd.Series(overwrite)
            overwrite_index = self.u.get_indexer(
                overwrite_limits.index)
            target_limit[overwrite_index] = overwrite_limits.values

        if label == 'lower':
            self.lower_limit = target_limit
        elif label == 'upper':
            self.upper_limit = target_limit

        return

    def set_lower(self, lower_limit, overwrite=None):
        self.set_limit('lower', lower_limit, overwrite)
        return

    def set_upper(self, upper_limit, overwrite=None):
        self.set_limit('upper', upper_limit, overwrite)
        return

    def set_benchmark(self, benchmark):
        self.benchmark = pd.Series(benchmark, dtype=float).reindex(
            self.u, fill_value=0)
        return

    def add_variation_penalty(self, init_w, alpha, l1_ratio=0.0, total_variation=0.0):
        if l1_ratio > 0:
            raise NotImplementedError(
                "currenly only second order variation penalty is supported")

        self.init_w = pd.Series(init_w).reindex(self.u, fill_value=0).values
        self.alpha_var = alpha
        self.l1_ratio_var = l1_ratio
        self.total_variation = total_variation
        return

    def set_categorical_exposure(self, exposure, category_weight=None):
        exposure_matrix = pd.get_dummies(exposure, prefix='', prefix_sep='').sort_index(
            axis=1).reindex(self.u, fill_value=0).astype(float)

        if category_weight is not None:
            exposure_matrix = exposure_matrix.mul(
                pd.Series(category_weight).reindex(exposure_matrix.columns), axis=1)

        self.cov_matrix = exposure_matrix.values @ exposure_matrix.values.T
        return

    def set_factor_covariance(self, cov_matrix):
        if cov_matrix.shape != (expect_shape := (self.u_size, self.u_size)):
            raise ValueError(
                f'expect a covariance matrix of shape {expect_shape}')

        self.cov_matrix = cov_matrix
        return

    def check_and_setup_problem(self):
        self.P = matrix(self.cov_matrix)
        self.q = matrix(-self.cov_matrix @ self.benchmark.values)

        if self.λ_var > 0.0:
            M_var = matrix(np.identity(self.u_size))
            self.P += self.λ_var * (1 - self.l1_ratio) * M_var.T * M_var
            self.q += -self.λ_var * (1 - self.l1_ratio) * M_var * matrix(self.init_w)

        I = matrix(np.identity(self.u_size))
        self.G = matrix([-I, I])
        self.h = matrix(np.hstack((-self.lower_limit, self.upper_limit)))

        self.A = matrix(1.0, (1, self.u_size))
        self.b = matrix([self.total_limit])  # weights summed to 1
        return

    def solve(self):
        self.check_and_setup_problem()
        return solvers.qp(self.P, self.q,
                            self.G, self.h, self.A, self.b)


if __name__ == '__main__':
    pass
