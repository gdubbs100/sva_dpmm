import torch
import torch.nn as nn
import torch.distributions as dist

import utils.plotting as plotting

from algorithms.sva import SVA

dpmm = SVA(
    learning_rate = 1.0e-3,
    alpha = 1.0,
    new_cluster_threshold= .99,
    prune_and_merge_freq=1000,
    prune_cluster_threshold = .01,
    merge_cluster_distance_threshold = 1.0e-5
)

## sample data
clusters = [
    dist.MultivariateNormal(
        torch.tensor([10*torch.cos(torch.tensor(i/.5)), 10*torch.sin(torch.tensor(i/.3))]),
        torch.diag(torch.ones(2,))
    ) for i in range(-3, 3)
]
N = 1000
raw_data = torch.stack([
    d.sample(torch.tensor([N])) for d in clusters
]).reshape(-1, 2) 

if __name__ == "__main__":
    dpmm.run(raw_data)

    phi = dpmm.phi
    clusters = torch.argmax(
        torch.exp(dist.MultivariateNormal(phi, dpmm.sigma).log_prob(raw_data.unsqueeze(1)) ), 
    axis=1)

    plotting.plot_clusters(
        data = raw_data.detach().numpy(),
        phi = phi.detach().numpy(),
        clusters=clusters.detach().numpy()
    )
