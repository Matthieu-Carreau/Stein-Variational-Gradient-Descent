import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, pdist, squareform
import scipy
from scipy.stats import gaussian_kde

from visualization import visu

def gaussian_kernel(points, h=None):
    dist = cdist(points, points)
    n, d = points.shape
    if h is None:
        h = np.median(dist) ** 2 / np.log(1 + n)

    ker = np.exp(- dist**2 / h)

    particles_reshaped = points[:, np.newaxis, :]
    diff_matrix = points[np.newaxis, :, :] - particles_reshaped
    dK = 2 * diff_matrix * ker[:, :, np.newaxis] / h

    return ker, dK, h

def matrices_for_Istein(x, gld, ker, h):

        init_dist = pdist(x)
        pairwise_dists = squareform(init_dist)

        # kernel matrix
        kernal_xj_xi = np.exp(- pairwise_dists ** 2 / h)

        # first derivative
        d0=np.matrix(x - x[:,np.newaxis])
        d1=kernal_xj_xi* (-1 /h)
        d_kernal_xj_xi=np.multiply(d0,d1)


        # second derivatives : nabla_{xi,xj} k(xi,x) = ( k(xi,x) + (xi-x)^2 * k(xi,x) ) * 2/h
        d1= kernal_xj_xi * 4 / h
        d2= np.multiply(pairwise_dists**2, kernal_xj_xi) * 8/(h**2)
        d2_kernal_xj_xi=(d1-d2)

        # scores of nabla log pi(xi)
        d_log_pi_xi=np.copy(gld)

        return kernal_xj_xi, d_kernal_xj_xi, d2_kernal_xj_xi, d_log_pi_xi

def computeIstein(kernal_xj_xi, d_kernal_xj_xi, d2_kernal_xj_xi, d_log_pi_xi):

        n = kernal_xj_xi.shape[0]
        d_log_pi_xj = d_log_pi_xi[:,np.newaxis]

        D = np.zeros((n,n))

        # first term with second derivative
        A = d2_kernal_xj_xi
        D+=A

        # cross terms
        b1= d_log_pi_xj + d_log_pi_xi
        b1=b1.reshape((n,n))
        B=np.multiply(b1,d_kernal_xj_xi)
        D+=B

        # last term
        dot_product_d_log_pi=np.multiply(d_log_pi_xi, d_log_pi_xj)
        dot_product_d_log_pi=dot_product_d_log_pi.reshape((n,n))
        C=np.multiply(dot_product_d_log_pi,kernal_xj_xi)
        D+=C

        istein=np.sum(D)/(n**2)
        return istein

class KLDivergence:
    def __init__(self, dimension, target, n_pts_dim=30, min_pt=-5, max_pt=5):
        self.dim = dimension
        self.target = target
        self.n_pts_dim = n_pts_dim
        self.min_pt = min_pt
        self.max_pt = max_pt
        self.cube_volume = ((max_pt - min_pt)/n_pts_dim) ** dimension

        # Create a grid of points
        coords = []
        for d in range(dimension):
            coords.append(np.linspace(min_pt, max_pt, n_pts_dim))
        grid_points = np.meshgrid(*coords)
        grid_points = np.moveaxis(np.array(grid_points), 0, -1)
        grid_points = grid_points.reshape(n_pts_dim**dimension, dimension)
        self.grid_points = grid_points

        # Compute the target density on the grid
        target_density_val = target.density(grid_points)
        target_density_val /= target_density_val.sum()
        target_density_val /= self.cube_volume
        self.target_density_val = target_density_val

    def __call__(self, points, h=None):
        # Estimate the density of mu at the grid points
        mu_density = gaussian_kde(points.T, bw_method=h)
        mu_val = mu_density.evaluate(self.grid_points.T)
        mu_val /= mu_val.sum() # Renormalize to avoid boundary effects
        mu_val /= self.cube_volume

        # Compute the KL integral
        pos_ind = mu_val > 0 # Avoid computing log(0), considering 0 * log(0) = 0
        KL = np.sum(mu_val[pos_ind] * np.log(mu_val[pos_ind] / self.target_density_val[pos_ind]))
        KL *= self.cube_volume
        return KL


class SVGD():
    def __init__(self, points, target):
        self.points = points
        self.target = target

    def optimize(self, n_iter, lr=0.01, h=None, display_step=1, lamb=5e-4):
        n, d = self.points.shape
        KL = list()
        grad_prob_list = list()
        grad_repulse_list = list()
        grad_tot_list = list()
        KL_exp = list()
        kl_div = KLDivergence(d, self.target)
        Istein = list()

        for iter in range(n_iter):
            ker, dK, h_ = gaussian_kernel(self.points, h=h)
            gld = self.target.grad_log_density(self.points)
            grad_prob = ker @ gld / n
            grad_repulse = dK.sum(axis=0) / n
            grad_tot = grad_prob + grad_repulse
            grad_prob_list.append(np.linalg.norm(grad_prob))
            grad_repulse_list.append(np.linalg.norm(grad_repulse))
            grad_tot_list.append(np.linalg.norm(grad_tot))
            # compute Istein
            if d==1:
                kernal_xj_xi, d_kernal_xj_xi, d2_kernal_xj_xi, d_log_pi_xi = matrices_for_Istein(self.points, gld, ker, h)
                istein=computeIstein(kernal_xj_xi, d_kernal_xj_xi, d2_kernal_xj_xi, d_log_pi_xi)
                Istein.append(istein)

            n_pts_x = n
            x_val = np.linspace(-5, 5, n_pts_x)
            xx = np.expand_dims(x_val, axis=1)

            density_values = self.target.density(xx)
            density_values = density_values / density_values.sum()

            kde = gaussian_kde(self.points.reshape(-1,))
            mu = kde.evaluate(x_val)
            mu = mu / mu.sum()
            KL.append(kl_div(self.points))

            if iter % display_step == 0 or iter == n_iter - 1:
                print("Iteration", iter, " h =", h_)
                visu(self.target.density,
                     x=self.points, #grad=target_distribution.grad_log_density(x))
                     grad=grad_tot,
                     grad_prob=grad_prob,
                     repulse=grad_repulse,
                     title="Iteration " + str(iter))

            self.points += grad_tot*lr

        t = np.arange(1, n_iter)
        lamb = np.min(np.log(KL[0]/KL[1:])/(2*t))
        t = np.arange(n_iter)
        KL_exp = KL[0] * np.exp(-2*t*lamb)

        plt.plot(range(n_iter), grad_repulse_list, label='Grad repulse', c='green')
        plt.plot(range(n_iter), grad_prob_list, label='Grad prob', c='blue')
        plt.plot(range(n_iter), grad_tot_list, label='Grad tot', c='red')
        plt.legend()
        plt.title('Evolution of gradients along iterations.')
        plt.show()
        plt.plot(range(n_iter), KL, label=r'$KL(\mu_t||\pi)$')
        print(f'lambda = {lamb}')
        plt.plot(range(n_iter), KL_exp, label=r'$\exp{-2t\lambda}KL(\mu_0||\pi)$')
        plt.legend()
        plt.title('Evolution of KL along iterations.')
        plt.show()
        plt.plot(range(n_iter), np.log(KL[0]/KL))
        plt.title(r'ln(KL($\mu_0$||$\pi$)/KL($\mu_t$||$\pi$)) as a function of $t$')
        plt.show()
        if d==1 :
            #plot IStein, value and average
            av_Istein = np.cumsum(Istein)/np.arange(1, len(Istein)+1)
            theoretical_rate = [(Istein[0])/(i+1) for i in range(len(Istein))]
            plt.loglog(range(n_iter), Istein)
            plt.title('Evolution of IStein along iterations')
            plt.show()
            plt.loglog(range(n_iter), av_Istein, label='Average Istein')
            plt.loglog(range(n_iter), theoretical_rate, label=r'1/n'+' rate')
            plt.title('Evolution of average-IStein along iterations')
            plt.legend()
            plt.show()