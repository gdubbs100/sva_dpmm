import math

import torch
import torch.nn as nn
import torch.distributions as dist
import numpy as np

import utils.plotting as plotting

from algorithms.sva import SVA

dpmm = SVA(
    alpha = 1,
    gamma= 0.6,
    new_cluster_threshold= .99,
    prune_and_merge_freq=1000,
    prune_cluster_threshold = 0.01,
    merge_cluster_distance_threshold = .1,
    lr_floor = 0.01
)

## sample data
clusters = [
    dist.MultivariateNormal(
        torch.tensor([
            20*torch.cos(torch.tensor(i*math.pi/5)), 
            20*torch.sin(torch.tensor(i*math.pi/5)), 
            # 10*torch.sin(torch.tensor(i*math.pi/2))
        ]),
        torch.diag(torch.ones(2,))
    ) for i in range(-5, 5)
]
N = 500
raw_data = torch.stack([
    d.sample(torch.tensor([N])) for d in clusters
]).reshape(-1, 2) 

if __name__ == "__main__":
    ## shuffle data
    shuffdex = np.array(range(raw_data.size(0)))
    np.random.shuffle(shuffdex)
    raw_data = raw_data[shuffdex,:]
    dpmm.run(raw_data)

    phi = dpmm.mixture_dist.parameters['loc']
    clusters = torch.argmax(
        torch.exp(dpmm.mixture_dist.log_prob(raw_data.unsqueeze(1)) ), 
    axis=1)

    plotting.plot_clusters(
        data = raw_data.detach().numpy(),
        phi = phi.detach().numpy(),
        clusters=clusters.detach().numpy()
    )
