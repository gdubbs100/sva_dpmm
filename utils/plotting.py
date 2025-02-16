import torch
import matplotlib.pyplot as plt

def plot_clusters(data, phi, clusters):

    plt.scatter(
        data[:,0],
        data[:,1],
        c = clusters
    )

    plt.scatter(
        phi[:, 0],
        phi[:, 1],
        label = 'Cluster Means',
        c = 'red'
    )

    plt.legend()
    plt.show();