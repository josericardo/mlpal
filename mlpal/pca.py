#!/usr/bin/env python

import numpy as np
import pylab as pl
from sklearn.decomposition import RandomizedPCA
from itertools import cycle

def plot_pca(X, y, output_file='pca_plot.png'):
    rpca = RandomizedPCA(n_components=2)
    X_pca = rpca.fit_transform(X)

    colors = ['b', 'g']
    markers = ['+', 'o']

    for i, c, m in zip(np.unique(y), cycle(colors), cycle(markers)):
        pl.scatter(X_pca[y == i, 0], X_pca[y == i, 1],
            c=c, marker=m, label=i, alpha=0.5)

    _ = pl.legend(loc='best') # legend will be displayed in the best position

    pl.title('Classes plotted according to the 2 principal components (RandomizedPCA)')
    pl.savefig(output_file)
