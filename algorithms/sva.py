import torch
import torch.nn as nn
from torch.nn.functional import l1_loss
import torch.distributions as dist
import numpy as np

import logging

## setup logger
logging.basicConfig(
     format="%(asctime)s - %(levelname)s - %(message)s",
     style="%",
     datefmt="%Y-%m-%d %H:%M",
     level=logging.INFO,
)
logger = logging.getLogger(__name__)

def calculate_rho(data, w, phi, sigma):
    h=torch.vstack([dist.MultivariateNormal(_phi, sigma).log_prob(data) for _phi in phi])
    log_weights = torch.log(w) + h.T # Compute log of weights * likelihood
    log_rho = log_weights - torch.logsumexp(log_weights, dim=1, keepdim=True)  # Normalize
    return torch.exp(log_rho).squeeze()

# 2. calculate cluster similarity
def calculate_cluster_similarity(rho, K):
    ## TODO: make similarity metric an arg?
    return (
        torch.tensor(
            [l1_loss(rho[i,:], rho[j,:], reduction='mean') if j > i else 0. 
            for i in range(K) for j in range(K)]
        )
        .view(K, K)
    )

# 3. identify those clusters to merge
def get_clusters_to_merge(cluster_similarity_matrix, similarity_threshold):
    mask = (cluster_similarity_matrix > 0) & (cluster_similarity_matrix < similarity_threshold)
    to_merge = torch.nonzero(mask, as_tuple=False)
    values_to_merge = cluster_similarity_matrix[mask]
    order = torch.argsort(values_to_merge, descending=False) # order clusters in order of most similar
    return to_merge[order]

# 4. calculate merge statistics
def get_merge_statistics(clusters_to_merge, phi, rho, K):
    merged = []
    all_to_prune = []
    merge_statistics = dict()

    for m in clusters_to_merge:
        if (m[0] not in merged) and (m[1] not in merged):
            all_to_prune.append(m[0].item()) # prune first idx
            # use second idx as update
            merge_statistics[m[1].item()] = {
                'phi': (phi[m[0]] + phi[m[1]]) / 2,
                'rho': rho[m[0]] + rho[m[1]]
            }
            merged.append(m[0])
            merged.append(m[1])
    
    # convert to boolean tensor
    all_to_prune = torch.tensor([i in all_to_prune for i in range(K)])
    return merge_statistics, all_to_prune

## apply merge, then apply prune
def apply_merge(merge_statistics, phi, w):
    phi = phi.detach() ## remove from computation graph temporarily
    for k, v in merge_statistics.items():
        phi[k] = v['phi']
        w[k] = v['rho']
    phi = phi.requires_grad_(True) # reattach to graph
    return phi, w

def prune(prune_idx: torch.Tensor, w: torch.Tensor, phi: torch.Tensor):
    w = w[~prune_idx]
    phi = phi[~prune_idx]
    K = w.size(0)
    return w, phi, K

def merge_and_prune(data, w, phi, rho, sigma, K, similarity_threshold):
    rho_prime = calculate_rho(data, w, phi, sigma)
    cluster_similarity_matrix = calculate_cluster_similarity(rho_prime, K)
    clusters_to_merge = get_clusters_to_merge(cluster_similarity_matrix, similarity_threshold)
    merge_statistics, all_to_prune = get_merge_statistics(clusters_to_merge, phi, w, K)
    phi, w = apply_merge(merge_statistics, phi, w)
    w, phi, K = prune(all_to_prune, w, phi)
    return w, phi, K

class SVA:

    def __init__(
        self,
        learning_rate: float,
        alpha: float,
        gamma: float,
        new_cluster_threshold: float,
        prune_and_merge_freq: int,
        prune_cluster_threshold: float,
        merge_cluster_distance_threshold: float
    ):
        self.learning_rate = learning_rate
        self.alpha = torch.tensor([alpha])
        self.gamma = gamma
        self.new_cluster_threshold = new_cluster_threshold

        self.prune_and_merge_freq = prune_and_merge_freq
        self.prune_cluster_threshold = prune_cluster_threshold
        self.merge_cluster_distance_threshold = merge_cluster_distance_threshold

        self.K = 1
        self.rho = torch.tensor([1.])
        self.w = torch.tensor([1.])

        self.sigma = torch.diag(torch.ones(2,))
        mu0 = torch.zeros((2,))
        self.base_dist = dist.MultivariateNormal(mu0, 100*torch.diag(torch.ones(2,)))
    
    def incremental_fit(self, x, idx):
        new_phi = torch.nn.Parameter(self.base_dist.sample())
        _phi = torch.vstack((self.phi, new_phi))
        _w = torch.concat((self.w, self.alpha))

        self.rho = calculate_rho(x, _w, _phi, self.sigma)

        if self.rho[self.K] > self.new_cluster_threshold:
            logger.info("Adding new cluster...")

            self.w += self.rho[:self.K]
            w_next = torch.tensor([self.rho[self.K]])
            self.w = torch.concat((self.w, w_next))

            self.phi = torch.vstack((self.phi, new_phi))
            self.K += 1
            logger.info("Added new cluster %d at: %s", self.K, new_phi)

        else:

            self.rho = self.rho[:self.K] / self.rho[:self.K].sum()
            self.w+= self.rho[:self.K]
        
        ## do the update
        # self.optimizer.zero_grad()
        n = 100
        d = np.min([n, (idx+1)**self.gamma])
        loss = (
            self.base_dist.log_prob(self.phi) +
            d * self.rho.detach() * (
                dist.MultivariateNormal(self.phi, self.sigma).log_prob(x)
            )
        ).mean()
   

        grad_phi = torch.autograd.grad(loss, self.phi,  retain_graph=True)[0]  # Compute gradient manually
        self.phi = self.phi + 1/d * grad_phi  # Apply gradient update
        # loss.backward()
        # self.optimizer.step()
        self.phi = self.phi.detach().requires_grad_(True) # prevent old graph accumulation

    ## TODO: create run function separate from this class
    def run(self, data):
        ## TODO: create options for initialising first cluster
        ## initialise cluster with first datapoint
        self.phi = torch.nn.Parameter(data[0, ...].unsqueeze(0))
        self.optimizer = torch.optim.Adam([self.phi], lr=self.learning_rate)
        # self.phi = torch.nn.Parameter(self.base_dist.sample().unsqueeze(0))
        logger.info("Initialising Phi: %s", self.phi.size())

        for idx, x in enumerate(data[1:, ...]):

            if (idx + 1) % self.prune_and_merge_freq == 0:
                to_prune = (self.w / self.w.sum()) < self.prune_cluster_threshold

                if any(to_prune):
                    self.w, self.phi, self.K = prune(to_prune, self.w, self.phi)
                    logger.info("Pruning to %s clusters", self.K)
                else:
                    logger.info("No clusters to prune...")
                
                # if self.phi.size(0) > 1:
                #     ## TODO: log some info about which clusters are being merged.
                #     logger.info("%d Clusters before merging...", self.K)
                #     self.w, self.phi, self.K = merge_and_prune(
                #         data[((idx+1)-self.prune_and_merge_freq):idx], self.w, self.phi, 
                #         self.rho, self.sigma, self.K,
                #         self.merge_cluster_distance_threshold
                #     )
                #     logger.info("%d Clusters remain after merge", self.K)

            self.incremental_fit(x, idx)

