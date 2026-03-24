import numpy as np
import matplotlib.pyplot as plt



class Sphere:
    def __init__(self, dimension, metric=None):
        self.dimension = dimension
        self.metric = metric


    def _project_to(self, X, zero_tol=1e-16):
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



    """
    def sample(self, n_samples, seed=None, dtype=np.float64, zero_tol=1e-8):
        
        #Sample n_samples points on the unit sphere S^{d-1}.
        #Returns: array of shape (n_samples, d)
        
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n_samples, self.dimension), dtype=dtype)
        X = self._project_to(X, zero_tol=zero_tol) 
        return X
    """

    def dist(self, S1, S2):
        """
        Distance between two points on a sphere under chosen metric.

        For metric == "Spherical":
            d(S1, S2) = arccos( <S1,S2> ), in radians
        For metric == "euclidean":
            d(S1, S2) = || S1 - S2 ||_2
        """
        S1 = self._project_to(S1)
        S2 = self._project_to(S2)

        if self.metric == 'spherical':
            s = 0.5 * np.linalg.norm(S1 - S2)
            s = np.clip(s, 0.0, 1.0)
            return 2.0 * np.arcsin(s)

        elif self.metric == 'euclidean':
            return np.linalg.norm(S1 - S2)

        elif self.metric is None:     
            raise ValueError("Metric is not set. Use set_metric('spherical' or 'Euclidean') first.")

        else:
            raise ValueError(f"Unknown metric: {self.metric}")

       
    # with sample diameter control
    def sample(self, n_samples, seed=0, tol=1e-16, diam=None):
        dim = self.dimension
        rng = np.random.default_rng(seed)

        def points_on_hemisphere(normal_vec, n_samples):
            normal_vec /= np.linalg.norm(normal_vec) 
            X = rng.normal(loc=normal_vec, size=(n_samples, dim)).astype(np.float64)
            X /= np.linalg.norm(X, axis=1, keepdims=True)

            # remove points on the boundary of the circle
            dots = (X @ normal_vec).astype(np.float64)
            bad = np.abs(dots) <= np.float64(tol)
            while np.any(bad):
                k = int(bad.sum())
                X_new = rng.normal(size=(k, dim)).astype(np.float64)
                X_new /= np.linalg.norm(X_new, axis=1, keepdims=True)
                dots_new = X_new @ normal_vec
                X[bad] = X_new
                dots[bad] = dots_new
                bad = np.abs(dots) <= np.float64(tol)

            # fold into hemisphere
            mask = dots < 0
            X[mask] *= -1.0
            return X

        #normal_vec = rng.normal(loc=normal_vec, size=dim).astype(np.float64)
        normal_vec = np.zeros(dim, dtype=np.float64)
        normal_vec[-1] = 1.0
        if diam is None:
            return points_on_hemisphere(normal_vec, n_samples)
            

        else:
            X = [points_on_hemisphere(normal_vec, 1)[0]] 

            while len(X) < n_samples:
                x_new = points_on_hemisphere(normal_vec, 1)[0] 
                if all(self.dist(x_new, x) < diam for x in X):
                    X.append(x_new)

            return np.vstack(X).astype(np.float64)    

    """  
    # with Variance control 
    def sample(self, n_samples, seed=0, tol=1e-16, scale=1.0):
        dim = self.dimension
        rng = np.random.default_rng(seed)
        normal_vec = rng.normal(scale= scale, size=dim).astype(np.float64)
        normal_vec /= np.linalg.norm(normal_vec) 
        X = rng.normal(loc=normal_vec, scale= scale, size=(n_samples, dim)).astype(np.float64)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
            
        return np.vstack(X).astype(np.float64)    
    
    """  
    def sample_sin(self, N, theta, r=1):
        phis = np.random.rand(N)*2*np.pi
        temp1 = np.random.rand(N**2)*theta
        temp2 = np.random.rand(N**2)*theta
        thetas = temp1[temp2<=np.sin(temp1)][:N]
        x = r * np.sin(thetas) * np.cos(phis)
        y = r * np.sin(thetas) * np.sin(phis)
        z = r * np.cos(thetas)
        #print(thetas.size)
        #plt.hist(thetas, bins=100, alpha=0.3)
        #plt.plot(np.linspace(0, theta, 1000), 6000*np.sin(np.linspace(0, theta, 1000)), color="red")
        #plt.show()
        return np.stack([x, y, z], axis=1)
    
    def sample_polar(self, n_samples, seed=0, diam=np.pi, r=1, dtype=np.float64):
        dim = self.dimension
        rng = np.random.default_rng(seed)
        if diam > np.pi:
            raise ValueError("diam must be less than pi")
        if dim != 3:
            raise ValueError("the algorithm only works for dim = 3 (for now!)")
        else:
            theta = rng.uniform(0.0, 2.0*np.pi, size=n_samples).astype(dtype)  
            phi =  rng.uniform(0.0, 0.5*diam, size=n_samples).astype(dtype)  
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)       


        return np.stack([x, y, z], axis=1).astype(dtype)
 




    def set_metric(self, Metric):
        metric = Metric.lower()
        if metric not in {"euclidean", "spherical"}:
            raise ValueError(f"Unknown metric: {Metric}")
        self.metric = metric



    def geodesic(self, x, y, t, List=False):
        """
        Geodesic between x and y on the sphere.

        Modes
        -----
        1) List=False and t is a scalar in [0,1]:
            returns g(t)

        2) List=True:
        - if t is an int n>=2: returns [g(0), g(1/(n-1)), ..., g(1)]
        - if t is array-like of scalars: returns [g(t_i)] for each t_i

        Metrics
        -------
        - Euclidean (projected): g(t) = proj((1-t)x + ty)
        - Spherical (great-circle / SLERP):
            Let theta = arccos(<x,y>).
            g(t) = proj( sin((1-t)theta)/sin(theta) * x + sin(t theta)/sin(theta) * y )
        """
        if self.metric is None:
            raise ValueError("Metric is not set. Use set_metric('euclidean' or 'spherical') first.")

        metric = str(self.metric).lower().strip()
        if metric not in {"euclidean", "spherical"}:
            raise ValueError(f"Unknown metric: {self.metric}")

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.shape != y.shape:
            raise ValueError("x and y must have the same shape.")

        # Ensure points are on the sphere (your class already has this)
        x = self._project_to(x)
        y = self._project_to(y)

        def g_at(tau: float):
            if not np.isfinite(tau):
                raise ValueError("t must be finite.")
            if tau < 0.0 or tau > 1.0:
                raise ValueError("t must be in [0,1].")

            if metric == "euclidean":
                z = (1.0 - tau) * x + tau * y
                return self._project_to(z)

            # metric == "spherical" 
            dot = float(np.dot(x, y))
            dot = np.clip(dot, -1.0, 1.0)
            theta = np.arccos(dot)

            # x and y almost identical -> geodesic is essentially constant
            if theta < 1e-8:
                return x.copy()

            # x and y nearly antipodal: SLERP becomes numerically unstable / not unique
            if np.pi - theta < 1e-16:
                # pick a deterministic orthogonal direction and rotate
                # (any great circle through x and -x is a geodesic; we choose one)
                idx = int(np.argmax(np.abs(x)))
                v = np.zeros_like(x)
                v[(idx + 1) % x.size] = 1.0
                u = v - np.dot(v, x) * x
                nu = np.linalg.norm(u)
                if nu < 1e-12:
                    # fallback: return x (rare degeneracy)
                    return x.copy()
                u = u / nu
                z = np.cos(np.pi * tau) * x + np.sin(np.pi * tau) * u
                return self._project_to(z)

            sin_theta = np.sin(theta)
            w1 = np.sin((1.0 - tau) * theta) / sin_theta
            w2 = np.sin(tau * theta) / sin_theta
            z = w1 * x + w2 * y
            return self._project_to(z)

        # ---- dispatch
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

        # List=False
        if t == 0:
            return x
        elif t == 1:
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
