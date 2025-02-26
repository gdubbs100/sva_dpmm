import torch
import torch.nn as nn
from torch.nn.functional import l1_loss
import torch.distributions as dist
import numpy as np

import logging

from utils.lr_scheduler import SVALRScheduler
from utils.training import (
    calculate_rho, calculate_cluster_distance,
    get_clusters_to_merge, get_merge_statistics,
    apply_merge, prune,
    merge_and_prune
)


## setup logger
logging.basicConfig(
     format="%(asctime)s - %(levelname)s - %(message)s",
     style="%",
     datefmt="%Y-%m-%d %H:%M",
     level=logging.INFO,
)
logger = logging.getLogger(__name__)

class SVA:

    def __init__(
        self,
        alpha: float,
        gamma: float,
        new_cluster_threshold: float,
        prune_and_merge_freq: int,
        prune_cluster_threshold: float,
        merge_cluster_distance_threshold: float,
        lr_floor: float = 0.01,
    ):
        self.alpha = torch.tensor([alpha])
        self.gamma = gamma
        self.lr_floor = lr_floor
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
        d = np.min([1/self.lr_floor, (idx+1)**self.gamma])
        loss = (
            self.base_dist.log_prob(self.phi) +
            d * self.rho.detach() * (
                dist.MultivariateNormal(self.phi, self.sigma).log_prob(x)
            )
        ).sum()

        grad_phi = torch.autograd.grad(loss, self.phi,  retain_graph=True)[0]  # Compute gradient manually
        self.phi = self.phi + 1/d * grad_phi  # Apply gradient update
        self.phi = self.phi.detach().requires_grad_(True) # prevent old graph accumulation

    ## TODO: create run function separate from this class
    def run(self, data):
        ## TODO: create options for initialising first cluster
        ## initialise cluster with first datapoint
        self.phi = torch.nn.Parameter(data[0, ...].unsqueeze(0))
        logger.info("Initialising Phi: %s", self.phi.size())
        for idx, x in enumerate(data[1:, ...]):
            self.incremental_fit(x, idx)

            if (idx + 1) % self.prune_and_merge_freq == 0:
                
                if self.phi.size(0) > 1:
                    ## TODO: log some info about which clusters are being merged.
                    logger.info("%d Clusters before merging...", self.K)
                    self.w, self.phi, self.K = merge_and_prune(
                        data[(idx+1 - self.prune_and_merge_freq):idx], self.w, self.phi, 
                        self.rho, self.sigma, self.K,
                        self.merge_cluster_distance_threshold
                    )
                    logger.info("%d Clusters remain after merge", self.K)
                
                    to_prune = (self.w / self.w.sum()) < self.prune_cluster_threshold

                    if any(to_prune):
                        self.w, self.phi, self.K = prune(to_prune, self.w, self.phi)
                        logger.info("Pruning to %s clusters", self.K)
                    else:
                        logger.info("No clusters to prune...")

            

