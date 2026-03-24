import numpy as np
from scipy.linalg import sqrtm, logm, inv, expm


class SymmetricPositiveDefinite:
    def __init__(self, dimension, metric=None):
        self.dimension = dimension
        self.metric = metric



    def _project_to(self, X, zero_tol=1e-8):
    
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


    def sample(self, n_samples, seed=None, dtype=np.float64, zero_tol=1e-8):
        rng = np.random.default_rng(seed)
        L = rng.standard_normal((n_samples, self.dimension, self.dimension), dtype=dtype)
        L = np.tril(L)  #lower-triangular matrices
        idx = np.arange(self.dimension)
        diag = np.abs(rng.standard_normal((n_samples, self.dimension), dtype=dtype)) + zero_tol
        L[:, idx, idx] = diag
        S = L @ np.transpose(L, (0, 2, 1)) #transpose each L_i in L
        return S
    
    def set_metric(self, Metric):
        metric = Metric.lower()
        if metric not in {"airm", "euclidean"}:
            raise ValueError(f"Unknown metric: {Metric}")
        self.metric = metric

    #todo: add projection onto the SPD
    
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
        


    def convex_combination(self, S1, S2, t: float):
        """
        Convex combination / geodesic interpolation between S1 and S2 at time t in [0,1].

        - For metric == 'euclidean':
              C(t) = (1 - t) * S1 + t * S2
        - For metric == 'airm':
              C(t) = S1^{1/2} exp( t * log( S1^{-1/2} S2 S1^{-1/2} ) ) S1^{1/2}
        """
        if self.metric is None:
            raise ValueError("Metric is not set. Use set_metric('AIRM' or 'Euclidean') first.")

        # ensure symmetry 
        S1 = 0.5 * (S1 + S1.T)
        S2 = 0.5 * (S2 + S2.T)

        if self.metric == "euclidean":
            C = (1.0 - t) * S1 + t * S2
            return 0.5 * (C + C.T)

        elif self.metric == "airm":
            S1_half = sqrtm(S1)
            S1_inv_half = inv(S1_half)
            X = S1_inv_half @ S2 @ S1_inv_half
            logX = logm(X)
            logX = np.real(logX)  # drop imaginary noise
            exp = expm(t * logX)
            C = S1_half @ exp @ S1_half
            return 0.5 * (C + C.T)

        else:
            raise ValueError(f"Unknown metric: {self.metric}")

        

    
    






