from geomstats.geometry.stratified.bhv_space import TreeSpace
from geomstats.geometry.stratified.bhv_space import BHVMetric
import numpy as np

class BHVSpace:

    def __init__(self, n_leaves):
            
            self.dimension = n_leaves
            self.space = TreeSpace(n_leaves=self.dimension)
            self.metric = self.space.metric

    def sample(self, n_samples):
        X = self.space.random_point(n_samples=n_samples)
        return X

    def _project_to(self, X):
        return X

    def dist(self, S1, S2):
        return self.metric.dist(S1, S2)

    def geodesic(self, x, y, t, List=False):

        geod = self.metric.geodesic(initial_point=x, end_point=y)

        def g_at(tau):
            if not (0.0 <= tau <= 1.0):
                raise ValueError("t must be in [0,1].")
            return geod(float(tau))

        if List:
            if isinstance(t, (int, np.integer)):
                n = int(t)
                if n < 2:
                    raise ValueError("n must be >= 2.")
                taus = np.linspace(0.0, 1.0, n)
                return [g_at(tau) for tau in taus]

            taus = np.asarray(t).ravel()
            return [g_at(float(tau)) for tau in taus]

        if np.isscalar(t):
            return g_at(float(t))

        raise ValueError("Invalid input for t.")
    
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



