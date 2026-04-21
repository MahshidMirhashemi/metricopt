from tqdm import tqdm
import numpy as np


class Prox:
    def __init__(self, space=None):
        self.space = space

    def set_space(self, space):
        self.space = space

    def prox_mapping_dist(self, x, xi, lam=1):
        if self.space is None:
            raise ValueError("Space not set. Use set_space(...) first.")

        tau = lam / (lam + 1)
        x  = self.space._project_to(x)
        xi = self.space._project_to(xi)

        z = self.space.geodesic(x, xi, t=tau, List=False)
        return z
    
    def cyclic(self, x, X, lam=0.5):
        X_new = []
        for xi in X:
            x = self.prox_mapping_dist(x, xi, lam)
            X_new.append(x)
        return X_new
    
    def relaxed_cyclic(self, x, X, tau= 0.5, lam=0.5):
        X_new = []
        for xi in X:
            #Id = np.eye(np.shape(x)[0])
            prox = self.prox_mapping_dist(x, xi, lam)
            x = self.space.geodesic(x, prox, tau, List=False)
            X_new.append(x)
        return X_new
    
    def cycle_circumference(self, S):
        length = 0.0
        for i in range(len(S)-1):
            length += self.space.dist(S[i], S[i+1])


        length += self.space.dist(S[-1], S[0])

        return length
    
    def Frechet_mean(self, x0, X, method = "relaxed", 
                            tau=0.5, 
                            lam=0.5,  
                            tol=1e-16, max_iter=200, show_progress= False):

        x_k = x0
        output = [list(X)]
        if method == "relaxed": 
            mapping = self.relaxed_cyclic
        else:
            mapping = self.cyclic

        for k in tqdm(range(max_iter), disable=not show_progress):
            X_next = mapping(x_k, X, tau, lam) # X_next is a list of the new cycle
            output.append(X_next)


            # stopping condition
            if self.space.dist(X_next[-1], x_k) <= tol:
                break
            x_k = X_next[-1]
        return x_k, output
    

    
