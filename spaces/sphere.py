import numpy as np


class Sphere:
    def __init__(self, dimension, metric=None):
        self.dimension = dimension
        self.metric = metric

    def _project_to(self, X, zero_tol=1e-8):
        """
        Project a vector or batch of vectors onto the unit sphere.
        X: shape (d,) or (n, d)
        """
        X = np.asarray(X)
        if X.ndim == 1:
            nrm = np.linalg.norm(X)
            if nrm < zero_tol:
                raise ValueError("Cannot project near-zero vector to sphere.")
            return X / nrm
        elif X.ndim == 2:
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            if np.any(norms < zero_tol):
                raise ValueError("Cannot project near-zero vector to sphere.")
            return X / norms
        else:
            raise ValueError("X must be 1D or 2D array.")

    def sample(
        self,
        n_samples,
        seed=None,
        dtype=np.float64,
        zero_tol=1e-8,
        distribution="uniform",
        mu=None,
        kappa=20):
        """
        Sample n_samples points on the unit sphere S^{d-1}.

        Parameters
        ----------
        n_samples : int
            Number of samples.
        seed : int or None
            Random seed.
        dtype : np.dtype
            Data type for the samples.
        zero_tol : float
            Tolerance for norm checks in projection.
        distribution : {"uniform", "vmf"}
            - "uniform": uniform distribution on the sphere
            - "vmf": von Mises-Fisher distribution with mean direction mu and concentration kappa
        mu : array-like, shape (d,), required if distribution == "vmf"
            Mean direction for vMF (will be normalized).
        kappa : float
            Concentration parameter for vMF. kappa = 0 -> uniform.

        Returns
        -------
        X : ndarray, shape (n_samples, d)
            Points on the unit sphere.
        """
        mu = np.zeros(self.dimension)
        mu[-1] = 1.0
        rng = np.random.default_rng(seed)
        d = self.dimension

        distribution = distribution.lower()
        if distribution == "uniform":
            X = rng.standard_normal((n_samples, d), dtype=dtype)
            X = self._project_to(X, zero_tol=zero_tol)
            return X

        elif distribution == "vmf":
            if mu is None:
                raise ValueError("mu must be provided for von Mises-Fisher sampling.")
            mu = np.asarray(mu, dtype=float)
            if mu.shape != (d,):
                raise ValueError(f"mu must have shape ({d},), got {mu.shape}.")
            # normalize mu
            mu_norm = np.linalg.norm(mu)
            if mu_norm < zero_tol:
                raise ValueError("mu must be non-zero.")
            mu = mu / mu_norm

            if kappa < 0:
                raise ValueError("kappa must be >= 0.")

            # special case: kappa == 0 -> uniform
            if kappa == 0:
                X = rng.standard_normal((n_samples, d), dtype=dtype)
                X = self._project_to(X, zero_tol=zero_tol)
                return X

            # ---- Wood's algorithm for vMF on S^{d-1} ----
            def _sample_w():
                b = (-2 * kappa + np.sqrt(4 * kappa**2 + (d - 1)**2)) / (d - 1)
                x0 = (1 - b) / (1 + b)
                c = kappa * x0 + (d - 1) * np.log(1 - x0**2)

                while True:
                    z = rng.beta((d - 1) / 2.0, (d - 1) / 2.0)
                    w = (1 - (1 + b) * z) / (1 - (1 - b) * z)
                    u = rng.uniform()
                    lhs = kappa * w + (d - 1) * np.log(1 - x0 * w)
                    if lhs - c >= np.log(u):
                        return w

            X = np.zeros((n_samples, d), dtype=float)
            e_d = np.zeros(d)
            e_d[-1] = 1.0

            for i in range(n_samples):
                w = _sample_w()
                # sample v ~ uniform on S^{d-2}
                v = rng.standard_normal(d - 1)
                v /= np.linalg.norm(v)

                x_ = np.zeros(d)
                x_[-1] = w
                factor = np.sqrt(1 - w**2)
                x_[:-1] = factor * v

                # rotate north pole e_d to mu via Householder
                if np.allclose(mu, e_d):
                    X[i] = x_
                else:
                    u = e_d - mu
                    u /= np.linalg.norm(u)
                    H = np.eye(d) - 2 * np.outer(u, u)
                    X[i] = H @ x_

            return X.astype(dtype)

        else:
            raise ValueError(f"Unknown distribution: {distribution!r}")

    """
    def sample(self, n_samples, seed=None, dtype=np.float64, zero_tol=1e-8):
        
        #Sample n_samples points uniformly on the unit sphere S^{d-1}.
        #Returns: array of shape (n_samples, d)
        
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n_samples, self.dimension), dtype=dtype)
        X = self._project_to(X, zero_tol=zero_tol) 
        return X
    """

    def set_metric(self, Metric):
        metric = Metric.lower()
        if metric not in {"euclidean", "spherical"}:
            raise ValueError(f"Unknown metric: {Metric}")
        self.metric = metric

    def dist(self, S1, S2):
        """
        Distance between two SPD matrices under chosen metric.

        For metric == "Spherical":
            d(S1, S2) = arccos( <S1,S2> ), in radians
        For metric == "euclidean":
            d(S1, S2) = || S1 - S2 ||_2
        """
        S1 = self._project_to(S1)
        S2 = self._project_to(S2)

        if self.metric == 'spherical':
            dot = float(np.dot(S1, S2))
            dot = np.clip(dot, -1.0, 1.0)
            return np.arccos(dot)

        elif self.metric == 'euclidean':
            return np.linalg.norm(S1 - S2)

        elif self.metric is None:     
            raise ValueError("Metric is not set. Use set_metric('AIRM' or 'Euclidean') first.")

        else:
            raise ValueError(f"Unknown metric: {self.metric}")
        


    def convex_combination(self, x, y, t: float):
        if self.metric is None:
            raise ValueError("Metric is not set. Use set_metric('euclidean' or 'geodesic') first.")

        x = self._project_to(x)
        y = self._project_to(y)

        if self.metric == "euclidean":
            z = (1.0 - t) * x + t * y
            return self._project_to(z)

        elif self.metric == "spherical":
            dot = float(np.dot(x, y))
            dot = np.clip(dot, -1.0, 1.0)
            theta = np.arccos(dot)

            if theta < 1e-8:
                # x and y almost identical → fallback to x
                return x.copy()

            sin_theta = np.sin(theta)
            w1 = np.sin((1.0 - t) * theta) / sin_theta
            w2 = np.sin(t * theta) / sin_theta
            z = w1 * x + w2 * y
            return self._project_to(z)

        else:
            raise ValueError(f"Unknown metric: {self.metric}")
