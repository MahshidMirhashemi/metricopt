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
    

    def Frechet_mean_EU(self, x0, X, method = "relaxed", 
                        tau=0.5, 
                        lam=0.5,  
                        tol=1e-16, max_iter=200, show_progress= False,
                        plot=False, geodesic_points=100):

        def plot_cycle_set(ax, Xset, alpha=1.0, title=None):

            X_arr = np.asarray(Xset, dtype=float)
            n = len(X_arr)

            for i in range(n):
                ax.scatter(X_arr[i][0], X_arr[i][1], color='blue', alpha=alpha, s=40)
                ax.annotate(str(i), (X_arr[i][0], X_arr[i][1]), xytext=(0, 6),
                            textcoords="offset points", ha="center", fontsize=9, alpha=alpha)

            for i in range(n - 1):
                G = self.space.geodesic(X_arr[i], X_arr[i+1], geodesic_points, List=True)
                G = np.asarray(G, dtype=float)
                ax.plot(G[:, 0], G[:, 1], color='gray', alpha=alpha, linewidth=1)

            if n > 1:
                G = self.space.geodesic(X_arr[-1], X_arr[0], geodesic_points, List=True)
                G = np.asarray(G, dtype=float)
                ax.plot(G[:, 0], G[:, 1], color='gray', alpha=alpha, linewidth=1)

            if title is not None:
                ax.set_title(title)

            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.axis("equal")
            ax.grid(True)
            
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

        if plot:
            import numpy as np
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 6))
            m = len(output)
            alphas = np.linspace(0.15, 1.0, m)

            for j, Xset in enumerate(output):
                plot_cycle_set(ax, Xset, alpha=alphas[j])

            avg = self.space.average(X)
            ax.scatter(avg[0], avg[1], color='red', s=80, marker='x', label='average')
            ax.legend()
            plt.show()

        return x_k, output

