import unittest
import numpy as np
from cvxopt import matrix, spdiag, mul, div, sqrt, normal, setseed
from cvxopt import blas, lapack, solvers, sparse, spmatrix
import math
from qopt import QOptimizer

def l1regls(A, b):
    """

    Returns the solution of l1-norm regularized least-squares problem

        minimize || A*x - b ||_2^2  + || x ||_1.

    """

    m, n = A.size
    q = matrix(1.0, (2*n, 1))
    q[:n] = -2.0 * A.T * b

    def P(u, v, alpha=1.0, beta=0.0):
        """
            v := alpha * 2.0 * [ A'*A, 0; 0, 0 ] * u + beta * v
        """
        v *= beta
        v[:n] += alpha * 2.0 * A.T * (A * u[:n])

    def G(u, v, alpha=1.0, beta=0.0, trans='N'):
        """
            v := alpha*[I, -I; -I, -I] * u + beta * v  (trans = 'N' or 'T')
        """

        v *= beta
        v[:n] += alpha*(u[:n] - u[n:])
        v[n:] += alpha*(-u[:n] - u[n:])

    h = matrix(0.0, (2*n, 1))

    # Customized solver for the KKT system
    #
    #     [  2.0*A'*A  0    I      -I     ] [x[:n] ]     [bx[:n] ]
    #     [  0         0   -I      -I     ] [x[n:] ]  =  [bx[n:] ].
    #     [  I        -I   -D1^-1   0     ] [zl[:n]]     [bzl[:n]]
    #     [ -I        -I    0      -D2^-1 ] [zl[n:]]     [bzl[n:]]
    #
    # where D1 = W['di'][:n]**2, D2 = W['di'][:n]**2.
    #
    # We first eliminate zl and x[n:]:
    #
    #     ( 2*A'*A + 4*D1*D2*(D1+D2)^-1 ) * x[:n] =
    #         bx[:n] - (D2-D1)*(D1+D2)^-1 * bx[n:] +
    #         D1 * ( I + (D2-D1)*(D1+D2)^-1 ) * bzl[:n] -
    #         D2 * ( I - (D2-D1)*(D1+D2)^-1 ) * bzl[n:]
    #
    #     x[n:] = (D1+D2)^-1 * ( bx[n:] - D1*bzl[:n]  - D2*bzl[n:] )
    #         - (D2-D1)*(D1+D2)^-1 * x[:n]
    #
    #     zl[:n] = D1 * ( x[:n] - x[n:] - bzl[:n] )
    #     zl[n:] = D2 * (-x[:n] - x[n:] - bzl[n:] ).
    #
    # The first equation has the form
    #
    #     (A'*A + D)*x[:n]  =  rhs
    #
    # and is equivalent to
    #
    #     [ D    A' ] [ x:n] ]  = [ rhs ]
    #     [ A   -I  ] [ v    ]    [ 0   ].
    #
    # It can be solved as
    #
    #     ( A*D^-1*A' + I ) * v = A * D^-1 * rhs
    #     x[:n] = D^-1 * ( rhs - A'*v ).

    S = matrix(0.0, (m, m))
    Asc = matrix(0.0, (m, n))
    v = matrix(0.0, (m, 1))

    def Fkkt(W):

        # Factor
        #
        #     S = A*D^-1*A' + I
        #
        # where D = 2*D1*D2*(D1+D2)^-1, D1 = d[:n]**-2, D2 = d[n:]**-2.

        d1, d2 = W['di'][:n]**2, W['di'][n:]**2

        # ds is square root of diagonal of D
        ds = math.sqrt(2.0) * div(mul(W['di'][:n], W['di'][n:]),
                                  sqrt(d1+d2))
        d3 = div(d2 - d1, d1 + d2)

        # Asc = A*diag(d)^-1/2
        Asc = A * spdiag(ds**-1)

        # S = I + A * D^-1 * A'
        blas.syrk(Asc, S)
        S[::m+1] += 1.0
        lapack.potrf(S)

        def g(x, y, z):

            x[:n] = 0.5 * (x[:n] - mul(d3, x[n:]) +
                           mul(d1, z[:n] + mul(d3, z[:n])) - mul(d2, z[n:] -
                                                                 mul(d3, z[n:])))
            x[:n] = div(x[:n], ds)

            # Solve
            #
            #     S * v = 0.5 * A * D^-1 * ( bx[:n] -
            #         (D2-D1)*(D1+D2)^-1 * bx[n:] +
            #         D1 * ( I + (D2-D1)*(D1+D2)^-1 ) * bzl[:n] -
            #         D2 * ( I - (D2-D1)*(D1+D2)^-1 ) * bzl[n:] )

            blas.gemv(Asc, x, v)
            lapack.potrs(S, v)

            # x[:n] = D^-1 * ( rhs - A'*v ).
            blas.gemv(Asc, v, x, alpha=-1.0, beta=1.0, trans='T')
            x[:n] = div(x[:n], ds)

            # x[n:] = (D1+D2)^-1 * ( bx[n:] - D1*bzl[:n]  - D2*bzl[n:] )
            #         - (D2-D1)*(D1+D2)^-1 * x[:n]
            x[n:] = div(x[n:] - mul(d1, z[:n]) - mul(d2, z[n:]), d1+d2)\
                - mul(d3, x[:n])

            # zl[:n] = D1^1/2 * (  x[:n] - x[n:] - bzl[:n] )
            # zl[n:] = D2^1/2 * ( -x[:n] - x[n:] - bzl[n:] ).
            z[:n] = mul(W['di'][:n],  x[:n] - x[n:] - z[:n])
            z[n:] = mul(W['di'][n:], -x[:n] - x[n:] - z[n:])

        return g
    A = matrix([[matrix(1.0, (1, n))], [matrix(0.0, (1, n))]])
    return solvers.coneqp(P, q, G, h, A, matrix([1.0]), kktsolver=Fkkt)['x'][:n]


def l1regls2(A, b):
    """

    Returns the solution of l1-norm regularized least-squares problem

        minimize || A*x - b ||_2^2  + || x ||_1.

    """

    m, n = A.size
    P = 2.0 * matrix([[A.T * A, matrix(0.0, (n, n))],
                     [matrix(0.0, (n, n)), matrix(0.0, (n, n))]])
    q = matrix(1.0, (2*n, 1))
    q[:n] = -2.0 * A.T * b

    G = matrix(np.vstack((np.hstack((np.identity(n), -np.identity(n))),
                          np.hstack((-np.identity(n), -np.identity(n))))))
    h = matrix(0.0, (2*n, 1))
    A = matrix([[matrix(1.0, (1, n))], [matrix(0.0, (1, n))]])
    return solvers.qp(P, q, G, h, A, matrix([1.0]))['x'][:n]


class TestQoptimizer(unittest.TestCase):

    def test_contraints(self):
        m, n = 20, 10
        P, q = normal(m, n), normal(m, 1)
        u = l1regls2(P, q)
        print(u)

        o = QOptimizer(range(0, n))
        o.set_regularization(1, l1_ratio=0.5)
        o.set_lower_bound(0.0)
        o.set_upper_bound(0.3)
        o.set_alpha(2 * np.array(P).T @ np.array(q))
        o.set_exposure(np.array(P))
        res = o.solve()['x'][:10]
        print(res)
        print(np.sum(res), np.max(res), np.min(res))
