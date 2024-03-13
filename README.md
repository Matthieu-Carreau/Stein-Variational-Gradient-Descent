# Stein-Variational-Gradient-Descent

Authors: Quentin Bourbon, Matthieu Carreau, Naël Farhan

Project for the MVA course Bayesian Machine Learning, study of the article:
Anna Korba, Adil Salim, Michael Arbel, Giulia Luise, and Arthur Gretton. A non-asymptotic
analysis for stein variational gradient descent, 2021

The src folder contains the code for our implementation and experiments. Three python files contain all the function used in our experiments, and "main.ipynb" contains the experiments.
We reimplemented the algorithm from scratch to perform these experiments, exept for two functions in the computation of the Stein Fischer information, that we reused from the code of Korba et al., available on their repository: https://github.com/akorba/SVGD_Non_Asymptotic .

To reproduce the experiments, one needs to clone this repository, install the following python libraries and run "main.ipynb":
- numpy
- matplotlib
- scipy
- seaborn
