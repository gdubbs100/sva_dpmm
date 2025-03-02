import torch
import torch.nn as nn
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
from utils.distributions import(
    DistDPMMParam,
    BaseDist,
    MixtureDPMM
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
        # mu0 = torch.zeros((2,))
        # self.base_dist = dist.MultivariateNormal(mu0, 100*torch.eye(2))
        mu0 = DistDPMMParam(
            name='loc', 
            value = dist.MultivariateNormal(torch.zeros(2,), 100*torch.eye(2))
        )
        sigma0 = DistDPMMParam(
            name='covariance_matrix', 
            # value = dist.Wishart(2, torch.eye(2))
            value = torch.eye(2)
        )
        self.base_dist = BaseDist(parameters=[mu0, sigma0])
    
    def incremental_fit(self, x, idx):
        ## sample from base distribution parameters
        # new_phi = torch.nn.Parameter(self.base_dist.sample())
        new_params = self.base_dist.sample()
        # breakpoint()
        self.mixture_dist.add_parameters(new_params)
        # _phi = torch.vstack((self.phi, new_phi))
        _w = torch.concat((self.w, self.alpha))

        ## calculate rho using DPMM mixture
        # self.rho = calculate_rho(x, _w, _phi, self.sigma)
        self.rho = calculate_rho(x, _w, self.mixture_dist)

        if self.rho[self.mixture_dist.K-1] > self.new_cluster_threshold:
            logger.info("Adding new cluster...")

            self.w += self.rho[:self.mixture_dist.K-1]
            w_next = torch.tensor([self.rho[self.mixture_dist.K-1]])
            self.w = torch.concat((self.w, w_next))

            # self.phi = torch.vstack((self.phi, new_phi))
            # self.K += 1
            logger.info("Added new cluster %d at: %s", self.mixture_dist.K+1, new_params)

        else:
            ## remove latest sampled parameter
            self.mixture_dist.remove_final_parameter()

            self.rho = self.rho[:(self.mixture_dist.K)] / self.rho[:(self.mixture_dist.K)].sum()
            self.w+= self.rho[:(self.mixture_dist.K)]
        
        ## do the update
        d = np.min([1/self.lr_floor, (idx+1)**self.gamma])
        # breakpoint()
        loss = (
            self.base_dist.log_prob(self.mixture_dist.parameters) +
            d * self.rho.detach() *(
                self.mixture_dist.log_prob(x)
            )
        ).sum()

        self.mixture_dist.update_learnable_parameters(
            loss=loss, 
            lr=1/d
        )

        # loss = (
        #     self.base_dist.log_prob(self.phi) +
        #     d * self.rho.detach() * (
        #         dist.MultivariateNormal(self.phi, self.sigma).log_prob(x)
        #     )
        # ).sum()

        # grad_phi = torch.autograd.grad(loss, self.phi,  retain_graph=True)[0]  # Compute gradient manually
        # self.phi = self.phi + 1/d * grad_phi  # Apply gradient update
        # self.phi = self.phi.detach().requires_grad_(True) # prevent old graph accumulation

    ## TODO: create run function separate from this class
    def run(self, data):
        ## TODO: create options for initialising first cluster
        ## initialise cluster with first datapoint
        # self.phi = torch.nn.Parameter(data[0, ...].unsqueeze(0))
        learnable_parameters = {'loc': True, 'covariance_matrix': False}
        init_parameters = {
            'loc': nn.Parameter(
                data[0, ...].unsqueeze(0), 
                requires_grad=learnable_parameters['loc']
            ),
            'covariance_matrix': nn.Parameter(
                torch.eye(2).unsqueeze(0),
                requires_grad = learnable_parameters['covariance_matrix']
            )
        }
        self.mixture_dist = MixtureDPMM(
            mixture_dist = dist.MultivariateNormal,
            parameters=init_parameters,
            learnable_parameters = learnable_parameters
        )
        logger.info("Initialising Phi: %s", data[0, ...].unsqueeze(0).size())
        for idx, x in enumerate(data[1:, ...]):
            self.incremental_fit(x, idx)

            # if (idx + 1) % self.prune_and_merge_freq == 0:
                
                # if self.phi.size(0) > 1:
                #     ## TODO: log some info about which clusters are being merged.
                #     logger.info("%d Clusters before merging...", self.K)
                #     self.w, self.phi, self.K = merge_and_prune(
                #         data[(idx+1 - self.prune_and_merge_freq):idx], self.w, self.phi, 
                #         self.rho, self.sigma, self.K,
                #         self.merge_cluster_distance_threshold
                #     )
                #     logger.info("%d Clusters remain after merge", self.K)
                
                #     to_prune = (self.w / self.w.sum()) < self.prune_cluster_threshold

                #     if any(to_prune):
                #         self.w, self.phi, self.K = prune(to_prune, self.w, self.phi)
                #         logger.info("Pruning to %s clusters", self.K)
                #     else:
                #         logger.info("No clusters to prune...")

            

