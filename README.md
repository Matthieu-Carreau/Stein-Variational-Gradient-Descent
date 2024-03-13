# Stein-Variational-Gradient-Descent

Authors: Quentin Bourbon, Matthieu Carreau, Naël Farhan

Project for the MVA course Bayesian Machine Learning, study of the article:
Anna Korba, Adil Salim, Michael Arbel, Giulia Luise, and Arthur Gretton. A non-asymptotic analysis for stein variational gradient descent, 2021

The src folder contains the code for our implementation and experiments. Three python files contain all the function used in our experiments, and "main.ipynb" contains the experiments.
We reimplemented the algorithm from scratch to perform these experiments, except for two functions in the computation of the Stein Fischer information, that we reused from the code of Korba et al., available on their repository (https://github.com/akorba/SVGD_Non_Asymptotic).

To reproduce the experiments, one needs to clone this repository, install the following python libraries and run "main.ipynb":
- numpy
- matplotlib
- scipy
- seaborn

The article consists in a theoretical analysis of the Stein Variational Gradient Descent (SVGD) in non-asymptotic cases. This algorithm treats the problem of sampling according to an arbitrary distribution as an optimization problem, using a point cloud that is iteratively steered towards the target distribution, as in the illustration below (from one of our experiments).

![evol_mu_2d_gaussian](https://github.com/Matthieu-Carreau/Stein-Variational-Gradient-Descent/assets/58993961/9822df5b-9aa2-48df-b205-e7c6721b68f0)
