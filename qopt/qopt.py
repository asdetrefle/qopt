
import numpy as np
import pandas as pd
from numbers import Real
from cvxopt import matrix, solvers
from .base import BaseOptimizer

solvers.options['abstol'] = 1e-8
solvers.options['reltol'] = 1e-7
solvers.options['show_progress'] = False


class QOptimizer(BaseOptimizer):
    """
    Mathematically, QOptimizer tries to solve the following problem:

    Maximize: a^T * h - λ_quad * (h - h_bmk)^T * V * (h - h_bmk)
                - λ_reg * (l1_ratio * |h - h0|_1 + (1 - l1_ratio) * |h - h0|_2^2)
                - OtherTerms
    Subject to: lower <= h <= upper,
                sum(h) = 1,
                OtherConstraints

    where h is the target weight vector
          h_bmk is the weight vector of the benchmark (e.g. index)
    """

    def __init__(self, index, **kwargs):
        self.index = pd.Index(index).drop_duplicates().sort_values()
        self.size = len(self.index)

        # objective
        self.alpha = None
        self.cov_matrix = None
        self.regularization = None  # (alpha, l1_ratio, base_value)
        self.has_l1_penalty = False

        # constraints
        self.lower_bound = np.zeros(self.size)
        self.upper_bound = np.ones(self.size)
        self.summed_value = float(kwargs.pop('summed_value', 1.0))

        if (benchmark := kwargs.pop('benchmark', 0.0)) is not None:
            self.set_benchmark(benchmark)

    def set_benchmark(self, benchmark):
        self.benchmark = pd.Series(benchmark, dtype=float).reindex(
            self.index, fill_value=0)
        return

    def set_alpha(self, alpha):
        """
        Add a regularization for the optimization problem.

        Parameters
        ----------
        alpha : list or ndarray or dict or pandas.Series
            alpha terms of the optimization problem
        """

        if isinstance(alpha, (dict, pd.Series)):
            alpha = pd.Series(alpha).reindex(self.index, fill_value=0).values

        if alpha.ndim == 1 and alpha.size == self.size:
            self.alpha = np.atleast_2d(alpha)
        elif alpha.ndim == 2 and alpha.shape == (self.size, 1):
            self.alpha = alpha
        else:
            raise ValueError(f"incompatible alpha array shape {alpha.shape}")
        return

    def set_bound(self, label, bound, overwrite=None):
        if isinstance(bound, Real):
            target_bound = np.full(self.size, bound)

            if overwrite is not None:
                overwrite_bound = pd.Series(overwrite)
                overwrite_index = self.index.get_indexer(
                    overwrite_bound.index)
                target_bound[overwrite_index] = overwrite_bound.values
        elif isinstance(bound, dict) or isinstance(bound, pd.Series):
            target_bound = pd.Series(bound).reindex(
                self.index, fill_value=0.0)
        else:
            assert len(bound) == self.size, \
                f"{label}_bound should be a scalar or an array of size of the index"
            target_bound = np.array(bound)

        target_bound = target_bound.astype(float)

        if label == 'lower':
            self.lower_bound = target_bound
        elif label == 'upper':
            self.upper_bound = target_bound
        return

    def set_lower_bound(self, lower_bound, overwrite=None):
        self.set_bound('lower', lower_bound, overwrite)
        return

    def set_upper_bound(self, upper_bound, overwrite=None):
        self.set_bound('upper', upper_bound, overwrite)
        return

    def set_regularization(self, lambda_, base_value=None, l1_ratio=0.0):
        """
        set the regularization for the optimization problem.

        Parameters
        ----------
        lambda_ : float
            Lambda constant that multiplies the penalty terms.
        base_value: float, dict or pandas.Series, default None
            The regularization base value which is the h0 in term |h - h0|.
            If base_value=None, then h0 ≡ 0 will be used
        l1_ratio: float, default=0.0
            The regularization mixing parameter, with 0 <= l1_ratio <= 1.
            For l1_ratio = 0 the penalty is an L2 penalty.
            For l1_ratio = 1 it is an L1 penalty.
            For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2.
        """

        assert lambda_ > 0, "regularization lambda must be strictly positive"

        if l1_ratio > 0:
            # raise NotImplementedError(
            #     "currenly only second order variation penalty is supported")
            self.has_l1_penalty = True

        if base_value is None:
            base_value = np.zeros(self.size)
        else:
            base_value = pd.Series(base_value).reindex(
                self.index, fill_value=0).values

        self.regularization = (lambda_, l1_ratio, base_value)
        return

    def set_categorical_exposure(self, category, category_weight=None):
        exposure = pd.get_dummies(category, prefix='', prefix_sep='').sort_index(
            axis=1).reindex(self.index, fill_value=0).astype(float)

        if category_weight is not None:
            exposure = exposure.mul(
                pd.Series(category_weight).reindex(exposure.columns), axis=1)

        self.cov_matrix = 2 * exposure.values @ exposure.values.T
        return

    def set_exposure(self, exposure):
        if exposure.shape[1] != self.size:
            raise ValueError(
                f'expect an exposure matrix of {self.size} columns')

        self.cov_matrix = 2 * exposure.T @ exposure
        return

    def set_factor_covariance(self, cov_matrix):
        if cov_matrix.shape != (expect_shape := (self.size, self.size)):
            raise ValueError(
                f'expect a covariance matrix of shape {expect_shape}')

        self.cov_matrix = cov_matrix
        return

    def setup_problem(self):
        self.P = matrix(self.cov_matrix)
        self.q = -matrix(self.cov_matrix @ self.benchmark.values)

        if self.alpha is not None:
            self.q -= matrix(self.alpha)

        I = matrix(np.identity(self.size))

        self.G = matrix([-I, I])
        self.h = matrix(np.hstack((-self.lower_bound, self.upper_bound)))

        self.A = matrix(np.ones((1, self.size)))
        self.b = matrix([self.summed_value])

        if self.regularization is not None:
            λ, l1_ratio, base_value = self.regularization

            self.P += 2 * λ * (1 - l1_ratio) * I.T * I
            self.q += -2 * λ * (1 - l1_ratio) * I * matrix(base_value)

            if l1_ratio > 0.0:
                # self.q += (matrix(base_value).T * self.P).T
                # self.h -= self.G * matrix(base_value)

                zero = matrix(0.0, (self.size, self.size))
                self.P = matrix([[self.P, zero],
                                 [zero, zero]])
                self.q = matrix([self.q, matrix(λ * l1_ratio, (self.size, 1))])
                self.G = matrix([[self.G, matrix([I, -I])],
                                 [matrix([zero, zero]), matrix([-I, -I])]])
                '''base_value'''
                self.h = matrix(
                    [self.h, matrix(base_value), -matrix(base_value)])
                self.A = matrix([self.A.T, matrix(0.0, (self.size, 1))]).T

        return

    def solve(self):
        if self.cov_matrix is None:
            raise ValueError("covariance matrix is not set!")

        self.setup_problem()
        return solvers.qp(self.P, self.q, self.G, self.h, self.A, self.b)


if __name__ == '__main__':
    pass
