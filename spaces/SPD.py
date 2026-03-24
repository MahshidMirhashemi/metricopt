import numpy as np
from scipy.linalg import sqrtm, logm, inv, expm

class SymmetricPositiveDefinite:
    def __init__(self, dimension, metric=None):
        self.dimension = dimension
        self.metric = metric.lower()


    def _project_to(self, X, zero_tol=1e-16):
    
        X = np.asarray(X)

        if X.ndim == 2:
            # single matrix
            M = 0.5 * (X + X.T)  # symmetrize
            eigvals, eigvecs = np.linalg.eigh(M)
            eigvals_clamped = np.maximum(eigvals, zero_tol)
            return eigvecs @ np.diag(eigvals_clamped) @ eigvecs.T

        elif X.ndim == 3:
            # batch of matrices
            n, d1, d2 = X.shape
            if d1 != d2:
                raise ValueError("Expected square matrices of shape (n, d, d).")
            out = np.empty_like(X, dtype=X.dtype)
            for i in range(n):
                M = 0.5 * (X[i] + X[i].T)
                eigvals, eigvecs = np.linalg.eigh(M)
                eigvals_clamped = np.maximum(eigvals, zero_tol)
                out[i] = eigvecs @ np.diag(eigvals_clamped) @ eigvecs.T
            return out

        else:
            raise ValueError("X must have shape (d, d) or (n, d, d).")

    """
    def sample(self, n_samples, seed=None, diam=None, dtype=np.float64, zero_tol=1e-8):
        if diam is None:
            np.random.seed(seed)
            L = np.random.random(size = (n_samples, self.dimension, self.dimension)).astype(dtype)
            #L = rng.uniform((n_samples, self.dimension, self.dimension), dtype=dtype)
            L = np.tril(L)  #lower-triangular matrices
            idx = np.arange(self.dimension)
            diag = np.abs(np.random.random((n_samples, self.dimension)).astype(dtype)) + zero_tol
            L[:, idx, idx] = diag
            S = L @ np.transpose(L, (0, 2, 1)) #transpose each L_i in L
            return S
        else:
            L = np.random.random(size = (1, self.dimension, self.dimension)).astype(dtype)

            while len(L) < n_samples:
                new_point = np.random.random(size = (self.dimension, self.dimension)).astype(dtype)
                for l in L:
                    if self.dist(l, new_point) <= diam:
                        L = np.concatenate([L, new_point[None, :, :]], axis=0)
            return             
    
    """

    def sample(self, n_samples, seed=None, diam=None, dtype=np.float64, tol=1e-16):
        if seed is not None:
            np.random.seed(seed)

        d = self.dimension
        idx = np.arange(d)

        def make_spd():
            L = np.random.random(size=(d, d)).astype(dtype)
            L = np.tril(L)
            L[idx, idx] = np.abs(np.random.random(size=d).astype(dtype)) + tol
            return L @ L.T

        if diam is None:
            # fast vectorized sampling
            L = np.random.random(size=(n_samples, d, d)).astype(dtype)
            L = np.tril(L)
            diag = np.abs(np.random.random(size=(n_samples, d)).astype(dtype)) + tol
            L[:, idx, idx] = diag
            S = L @ np.transpose(L, (0, 2, 1))
            return S

        # diam constraint sampling
        S_list = [make_spd()]
        while len(S_list) < n_samples:
            cand = make_spd()
            # accept if it's within diam of at least one existing sample
            if all(self.dist(Sj, cand) <= diam for Sj in S_list):
                S_list.append(cand)

        return np.stack(S_list, axis=0)

    def set_metric(self, Metric):
        metric = Metric.lower()
        if metric not in {"airm", "euclidean"}:
            raise ValueError(f"Unknown metric: {Metric}")
        self.metric = metric


    def dist(self, S1, S2):
        """
        Distance between two SPD matrices under chosen metric.

        For metric == "AIRM":
            d(S1, S2) = || log( S1^{-1/2} S2 S1^{-1/2} ) ||_F
        For metric == "euclidean":
            d(S1, S2) = || S1 - S2 ||_F
        """
        if self.metric == 'airm':
            S1_inv_half = sqrtm(inv(S1))
            X = S1_inv_half @ S2 @ S1_inv_half
            log_X = np.real(logm(X))
            return np.linalg.norm(log_X, 'fro')

        elif self.metric == 'euclidean':
            return np.linalg.norm(S1 - S2, 'fro')

        elif self.metric is None:     
            raise ValueError("Metric is not set. Use set_metric('AIRM' or 'Euclidean') first.")

        else:
            raise ValueError(f"Unknown metric: {self.metric}")
        


    def geodesic(self, x, y, t, List=False):
        """
        Geodesic between x and y.

        Modes
        -----
        1) List=False and t is a scalar in [0,1]:
            returns g(t)

        2) List=True:
        - if t is an int n>=2: returns [g(0), g(1/(n-1)), ..., g(1)]
        - if t is array-like of scalars: returns [g(t_i)] for each t_i

        Metrics
        -------
        - Euclidean: g(t) = (1 - t) x + t y
        - AIRM:      g(t) = x^{1/2} exp(t log(x^{-1/2} y x^{-1/2})) x^{1/2}
        """
        if self.metric is None:
            raise ValueError("Metric is not set. Use set_metric('AIRM' or 'Euclidean') first.")

        metric = self.metric
        if metric not in {"euclidean", "airm"}:
            raise ValueError(f"Unknown metric: {self.metric}")

        x = np.asarray(x)
        y = np.asarray(y)
        if x.shape != y.shape:
            raise ValueError("x and y must have the same shape.")

        def sym(A):
            return 0.5 * (A + A.T)

        x = sym(x)
        y = sym(y)

        def g_at(tau: float):
            if not np.isfinite(tau):
                raise ValueError("t must be finite.")
            if tau < 0.0 or tau > 1.0:
                raise ValueError("t must be in [0,1].")

            if metric == "euclidean":
                C = (1.0 - tau) * x + tau * y
                return sym(C)

            # metric == "airm"
            x_half = np.real_if_close(sqrtm(x))
            x_inv_half = np.linalg.inv(x_half)

            X = sym(x_inv_half @ y @ x_inv_half)
            L = np.real_if_close(logm(X))
            E = np.real_if_close(expm(tau * L))

            C = x_half @ E @ x_half
            return sym(np.real_if_close(C))


        if List:
            # t is number of samples
            if isinstance(t, (int, np.integer)):
                n = int(t)
                taus = np.linspace(0.0, 1.0, n)
                return [g_at(float(tau)) for tau in taus]

            # t is iterable of parameters
            taus = np.asarray(t).ravel()
            if taus.size == 0:
                raise ValueError("Empty list/array of t values.")
            return [g_at(float(tau)) for tau in taus]

        # List=False: t must be a scalar
        if t==0:
            return x
        elif t==1:
            return y
        if not np.isscalar(t):
            raise ValueError("When List=False, t must be a scalar in [0,1].")
        return g_at(float(t))
    

    def diameter(self,points):
        """
        points: list/array of sample points
        returns: dmax, (i, j)
        """
        n = len(points)
        if n < 2:
            raise ValueError("Need at least 2 points to define a diameter.")

        dmax = -np.inf
        imax = jmax = None

        for i in range(n - 1):
            for j in range(i + 1, n):
                d = float(self.dist(points[i], points[j]))
                if d > dmax:
                    dmax = d
                    imax, jmax = i, j

        return dmax, (imax, jmax)




    
    






