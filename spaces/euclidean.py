import numpy as np


class EuclideanSpace:
    def __init__(self, dimension, metric=None):
        """
        Euclidean space R^d or R^{n x n}.

        Parameters
        ----------
        dimension : int or tuple
            If int, points have shape (dimension,)
            If tuple, points have shape dimension, e.g. (n, n)
        """
        if isinstance(dimension, int):
            if dimension <= 0:
                raise ValueError("dimension must be positive.")
            self.shape = (dimension,)
        elif isinstance(dimension, tuple):
            if len(dimension) == 0 or any(k <= 0 for k in dimension):
                raise ValueError("All entries of dimension must be positive.")
            self.shape = tuple(dimension)
        else:
            raise TypeError("dimension must be int or tuple.")

        self.dimension = dimension
        self.metric = "euclidean" if metric is None else metric.lower()
        if self.metric != "euclidean":
            raise ValueError("EuclideanSpace only supports the Euclidean metric.")

    def _project_to(self, X):
        """
        Identity map, kept only for API similarity with Sphere.
        Accepts one point or a batch of points.
        """
        X = np.asarray(X, dtype=np.float64)

        if X.shape == self.shape:
            return X
        elif X.ndim >= 2 and X.shape[1:] == self.shape:
            return X
        else:
            raise ValueError(
                f"Input must have shape {self.shape} or (n_samples, {self.shape}), got {X.shape}."
            )

    def set_metric(self, Metric):
        metric = Metric.lower()
        if metric != "euclidean":
            raise ValueError(f"Unknown metric: {Metric}")
        self.metric = metric

    def dist(self, X, Y):
        """
        Euclidean distance (or Frobenius norm for matrices).
        """
        X = self._project_to(X)
        Y = self._project_to(Y)

        if X.shape != self.shape or Y.shape != self.shape:
            raise ValueError(f"Points must each have shape {self.shape}.")

        return np.linalg.norm(X - Y)

    def sample(self, n_samples, seed=0, diam=1.0, center=None, dtype=np.float64):
        """
        Uniform sampling in a Euclidean ball.

        If center is None, sample in the ball centered at the origin.
        If diam is given, points are sampled uniformly in the ball of radius diam/2,
        hence every pair has distance <= diam.

        Parameters
        ----------
        n_samples : int
        seed : int
        diam : float
            Desired diameter bound for the sample cloud.
        center : array-like or None
            Center of the ball.
        dtype : numpy dtype

        Returns
        -------
        X : ndarray of shape (n_samples, *self.shape)
        """
        if diam <= 0:
            raise ValueError("diam must be positive.")

        rng = np.random.default_rng(seed)
        radius = 0.5 * diam
        flat_dim = int(np.prod(self.shape))

        if center is None:
            center = np.zeros(self.shape, dtype=dtype)
        else:
            center = np.asarray(center, dtype=dtype)
            if center.shape != self.shape:
                raise ValueError(f"center must have shape {self.shape}, got {center.shape}")

        X = np.empty((n_samples,) + self.shape, dtype=dtype)

        for i in range(n_samples):
            # random direction
            v = rng.normal(size=flat_dim).astype(dtype)
            nv = np.linalg.norm(v)
            while nv < 1e-16:
                v = rng.normal(size=flat_dim).astype(dtype)
                nv = np.linalg.norm(v)
            v /= nv

            # correct radial law for uniform-in-ball sampling
            u = rng.random()
            r = radius * (u ** (1.0 / flat_dim))

            X[i] = center + (r * v).reshape(self.shape)

        return X.astype(dtype)

    def geodesic(self, x, y, t, List=False):
        """
        Euclidean geodesic = straight line interpolation.
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if x.shape != self.shape or y.shape != self.shape:
            raise ValueError(f"x and y must have shape {self.shape}.")

        def g_at(tau):
            if not np.isfinite(tau):
                raise ValueError("t must be finite.")
            if tau < 0.0 or tau > 1.0:
                raise ValueError("t must be in [0,1].")
            return (1.0 - tau) * x + tau * y

        if List:
            if isinstance(t, (int, np.integer)):
                n = int(t)
                if n < 2:
                    raise ValueError("When List=True and t is int, it must be n>=2.")
                taus = np.linspace(0.0, 1.0, n)
                return [g_at(float(tau)) for tau in taus]

            taus = np.asarray(t).ravel()
            if taus.size == 0:
                raise ValueError("Empty list/array of t values.")
            return [g_at(float(tau)) for tau in taus]

        if not np.isscalar(t):
            raise ValueError("When List=False, t must be a scalar in [0,1].")

        return g_at(float(t))

    def diameter(self, points):
        """
        points: array of shape (n_samples, *self.shape)
        returns: dmax, (i, j)
        """
        points = np.asarray(points, dtype=np.float64)
        if points.ndim < 2 or points.shape[1:] != self.shape:
            raise ValueError(
                f"points must have shape (n_samples, {self.shape}), got {points.shape}"
            )

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
    
    def average(self, points, weights=None):
        """
        Compute the average of a sample in Euclidean space.

        Parameters
        ----------
        points : ndarray
            Array of shape (n_samples, *self.shape)
        weights : array-like or None
            Optional nonnegative weights of length n_samples.
            If None, the usual arithmetic mean is returned.

        Returns
        -------
        avg : ndarray of shape self.shape
        """
        points = np.asarray(points, dtype=np.float64)

        if points.ndim < 2 or points.shape[1:] != self.shape:
            raise ValueError(
                f"points must have shape (n_samples, {self.shape}), got {points.shape}"
            )

        n = len(points)
        if n == 0:
            raise ValueError("Cannot average an empty sample.")

        if weights is None:
            return np.mean(points, axis=0)

        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if len(weights) != n:
            raise ValueError("weights must have length equal to number of sample points.")
        if np.any(weights < 0):
            raise ValueError("weights must be nonnegative.")

        wsum = np.sum(weights)
        if wsum <= 0:
            raise ValueError("weights must sum to a positive number.")

        weights = weights / wsum
        return np.tensordot(weights, points, axes=(0, 0))