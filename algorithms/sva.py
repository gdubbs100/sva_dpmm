import torch
import torch.nn as nn
import torch.distributions as dist
import numpy as np

import logging

from utils.lr_scheduler import SVALRScheduler
from utils.training import (
    calculate_rho, calculate_cluster_distance,
    get_clusters_to_merge, get_merge_statistics,
    merge_and_prune_weights, prune_weights
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

        # self.K = 1
        self.rho = torch.tensor([1.])
        self.w = torch.tensor([1.])

        ## TODO: perhaps refactor to make base-dist an arg
        ## create base dist
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
        new_params = self.base_dist.sample()
        self.mixture_dist.add_parameters(new_params)
        _w = torch.concat((self.w, self.alpha))

        ## calculate rho using DPMM mixture
        self.rho = calculate_rho(x, _w, self.mixture_dist)

        if self.rho[self.mixture_dist.K-1] > self.new_cluster_threshold:
            logger.info("Adding new cluster...")

            self.w += self.rho[:self.mixture_dist.K-1]
            w_next = torch.tensor([self.rho[self.mixture_dist.K-1]])
            self.w = torch.concat((self.w, w_next))

            logger.info("Added new cluster %d at: %s", self.mixture_dist.K+1, new_params)

        else:
            ## remove latest sampled parameter
            self.mixture_dist.remove_final_parameter()

            self.rho = self.rho[:(self.mixture_dist.K)] / self.rho[:(self.mixture_dist.K)].sum()
            self.w += self.rho[:(self.mixture_dist.K)]
        
        ## do the update
        d = np.min([1/self.lr_floor, (idx+1)**self.gamma])

        loss = (
            self.base_dist.log_prob(self.mixture_dist.parameters) +
            d * self.rho.detach() * self.mixture_dist.log_prob(x)
        ).sum()

        self.mixture_dist.update_learnable_parameters(
            loss=loss, 
            lr=1/d
        )

    ## TODO: create run function separate from this class
    def run(self, data):
        ## TODO: create options for passing mixture dist to run
        ## TODO: create options for initialising first cluster
        ## initialise cluster with first datapoint
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

            if (idx + 1) % self.prune_and_merge_freq == 0:
                
                if self.mixture_dist.K > 1:
                    ## TODO: log some info about which clusters are being merged.
                    logger.info("%d Clusters before merging...", self.mixture_dist.K)

                    merge_mapping, merged_clusters = get_merge_statistics(
                        data[(idx+1 - self.prune_and_merge_freq):idx],
                        self.w, 
                        self.mixture_dist, 
                        self.merge_cluster_distance_threshold
                    )
                    ## apply the merge / prune
                    self.mixture_dist.merge_parameters(merge_mapping)
                    self.mixture_dist.remove_parameters(merged_clusters)
                    self.w = merge_and_prune_weights(merge_mapping, self.w)
                    logger.info("%d Clusters remain after merge", self.mixture_dist.K)
                
                    to_prune = (self.w / self.w.sum()) < self.prune_cluster_threshold

                    if any(to_prune):
                        self.w = prune_weights(self.w, to_prune)
                        self.mixture_dist.remove_parameters(to_prune)
                        logger.info("Pruning to %s clusters", self.mixture_dist.K)
                    else:
                        logger.info("No clusters to prune...")

            

