from tqdm import tqdm


class Prox:
    def __init__(self, space=None):
        self.space = space

    def set_space(self, space):
        self.space = space


    def prox_mapping_dist(self, x, xi, lam=0.5):
        if self.space is None:
            raise ValueError("Space not set. Use set_space(...) first.")

        tau = lam / (lam + 1)
        x  = self.space._project_to(x)
        xi = self.space._project_to(xi)

        z = self.space.convex_combination(x, xi, t=tau)
        return z
    
    def cyclic(self, x, X, lam=0.5):
        for xi in X:
            x = self.prox_mapping_dist(x, xi, lam)
        return x
    

    def relaxed_cyclic(self, x, X, tau= 0.5, lam=0.5):
        for xi in X:
            #Id = np.eye(np.shape(x)[0])
            prox = self.prox_mapping_dist(x, xi, lam)
            x = self.space.convex_combination(x, prox, tau)
        return x
    

    def Frechet_mean(self, x0, X, method = "relaxed", 
                            tau=0.5, 
                            lam=0.5,  
                            p=2.0,
                            tol=1e-16, max_outer_iter=200):

        x_k = x0
        output = [x_k]
        if method == "relaxed": 
            mapping = self.relaxed_cyclic
        else:
            mapping = self.cyclic

        for k in tqdm(range(max_outer_iter)):
            x_next = mapping(x_k, X, tau, lam)
            output.append(x_next)


            # stopping condition
            if self.space.dist(x_next, x_k) <= tol:
                break
            x_k = x_next
        return x_k, output
