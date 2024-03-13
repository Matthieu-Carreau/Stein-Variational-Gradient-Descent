import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def visu_1d(target_density, x=None, grad=None, grad_prob=None, repulse=None,
            x_lim=(-5, 5), n_pts_x=200, title=""):
    """
    Visualization of the target density in 1 dimension, with the particles and their gradients
    """
    n_pts_x = x.shape[0]
    # Create the grid of values
    x_val = np.linspace(x_lim[0], x_lim[1], n_pts_x)
    xx = np.expand_dims(x_val, axis=1)

    # Compute the density
    density_values = target_density(xx)
    #density_values = density_values.reshape(n_pts_x, 1)
    density_values = density_values / density_values.sum()

    # Plot the density
    plt.plot(x_val, density_values, label="Target distribution")
    plt.xlabel("X")
    plt.ylabel("Density")
    plt.title(title)

    if x is not None:

        # Plot the points
        plt.scatter(x, np.zeros_like(x), color="black", marker=".")

        kde = gaussian_kde(x.reshape(-1,))
        mu = kde.evaluate(x_val)
        mu = mu / mu.sum()
        plt.plot(x_val, mu, label="Current distribution")

        if grad_prob is not None:
            grad_plot = grad_prob
            plt.quiver(x.flatten(), np.zeros_like(x), grad_plot.flatten(), np.zeros_like(grad_plot), color="blue", scale_units="xy", angles="xy", scale=1, width=0.005)

        if repulse is not None:
            grad_plot = repulse
            plt.quiver(x.flatten(), np.zeros_like(x), grad_plot.flatten(), np.zeros_like(grad_plot), color="green", scale_units="xy", angles="xy", scale=1, width=0.005)


        if grad is not None:
            grad_plot = grad
            plt.quiver(x.flatten(), np.zeros_like(x), grad_plot.flatten(), np.zeros_like(grad_plot), color="red", scale_units="xy", angles="xy", scale=1, width=0.005)

    plt.legend()
    plt.show()

def visu_2d(target_density, x=None, grad=None, grad_prob=None, repulse=None,
            x_lim=(-5, 5), n_pts_x=101, title=""):
    """
    Visualization of the target density in 2 dimensions, with the particles and their gradients
    """
    x1_lim = x_lim
    x2_lim = x_lim

    n_pts_x1 = n_pts_x
    n_pts_x2 = n_pts_x

    # Create the grid of values
    x1_val = np.linspace(x1_lim[0], x1_lim[1], n_pts_x1)
    x2_val = np.linspace(x2_lim[0], x2_lim[1], n_pts_x2)

    x1, x2 = np.meshgrid(x1_val, x2_val)
    xx = np.concatenate((np.expand_dims(x1, axis=2), np.expand_dims(x2, axis=2)), axis=2)

    # Compute the density
    density_values = target_density(xx.reshape(-1, 2))
    density_values = density_values.reshape(n_pts_x2, n_pts_x1)

    # Plot the density
    x1_labels = np.round(np.linspace(x1_lim[0], x1_lim[1], 11), 1)
    x1_ticks = np.linspace(0, n_pts_x1-1, 11, dtype='int')
    x2_labels = np.round(np.linspace(x2_lim[0], x2_lim[1], 11), 1)
    x2_ticks = np.linspace(0, n_pts_x2-1, 11, dtype='int')

    plt.imshow(density_values, origin='lower')
    plt.xticks(ticks=x1_ticks, labels=x1_labels)
    plt.yticks(ticks=x2_ticks, labels=x2_labels)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title(title)
    plt.colorbar()


    if x is not None:
        scale = np.diag([(n_pts_x1-1) / (x1_lim[1] - x1_lim[0]), (n_pts_x2-1) / (x2_lim[1] - x2_lim[0])])
        x_plot = (x -np.array([x1_lim[0], x2_lim[0]])) @ scale

        # Plot the gradients
        if grad_prob is not None:
            grad_plot = grad_prob @ scale
            for pt, pt_grad in zip(x_plot, grad_plot):
                plt.arrow(pt[0], pt[1], pt_grad[0], pt_grad[1], color="blue", head_width=1, head_length=1)

        if repulse is not None:
            grad_plot = repulse @ scale
            for pt, pt_grad in zip(x_plot, grad_plot):
                plt.arrow(pt[0], pt[1], pt_grad[0], pt_grad[1], color="green", head_width=1, head_length=1)

        if grad is not None:
            grad_plot = grad @ scale
            for pt, pt_grad in zip(x_plot, grad_plot):
                plt.arrow(pt[0], pt[1], pt_grad[0], pt_grad[1], color="red", head_width=1, head_length=1)

        # Plot the points
        plt.scatter(x_plot[:, 0], x_plot[:, 1], color="teal", marker=".")


    plt.show()

def visu(target_density, x=None, grad=None, grad_prob=None, repulse=None,
         x_lim=(-5, 5), n_pts_x=101, title=""):
    """
    Visualization of the target density in 1 or 2 dimensions
    """
    d = x.shape[-1]
    if d == 1:
      visu_1d(target_density, x=x, grad=grad, grad_prob=grad_prob, repulse=repulse,
              x_lim=x_lim, n_pts_x=n_pts_x, title=title)
      return
    if d == 2:
      visu_2d(target_density, x=x, grad=grad, grad_prob=grad_prob, repulse=repulse,
              x_lim=x_lim, n_pts_x=n_pts_x, title=title)
      return