import torch
import torch.nn as nn
import torch.distributions as dist

from typing import Dict, Tuple
class DistDPMMParam:

    def __init__(
        self, 
        name: str,
        value: torch.distributions.Distribution | torch.Tensor,
        ):
        self.name = name
        self.value = value
    
    def get_value(self):
        if isinstance(self.value, torch.distributions.Distribution):
            return self.value.sample()
        else:
            return self.value
    
    def log_prob(self, x):
        if isinstance(self.value, torch.distributions.Distribution):
            return self.value.log_prob(x)
        else:
            ## NOTE: this works nicely for our purposes, but is this a sensible output?
            return torch.zeros(x.size(0))

class BaseDist:

    def __init__(self, parameters):
        self.parameters = parameters
    
    def sample(self):
        # breakpoint()
        return {
            param.name: param.get_value().unsqueeze(0) for param in self.parameters
        }
    
    def log_prob(self, parameter_values):
        # breakpoint()
        return torch.stack(
            [param.log_prob(parameter_values[param.name])
            for param in self.parameters]
        ).sum(axis=0)

class MixtureDPMM:

    def __init__(self, 
        mixture_dist: torch.distributions.Distribution, 
        parameters: Dict[str, torch.Tensor],
        learnable_parameters: Dict[str, bool]
        ):
        self.mixture_dist = mixture_dist # the type of distribution used
        self.parameters = parameters # dictionary of parameter values
        self.learnable_parameters = learnable_parameters # which parameters to keep gradients for
        self.K = 1

    def log_prob(self, x):
        # breakpoint()
        return self.mixture_dist(**self.parameters).log_prob(x)

    def add_parameters(self, new_params):
        # breakpoint()
        for param_name, param_values in new_params.items():
            self.parameters[param_name] = torch.vstack(
                (
                    self.parameters[param_name],
                    nn.Parameter(
                        param_values,#.reshape(self.parameters[param_name].size(0), -1), 
                        requires_grad=self.learnable_parameters[param_name]), 
                    )
            )
        self.K += 1
    
    def remove_parameters(self, remove_mask):
        for param_name in self.parameters.keys():
            self.parameters[param_name] = self.parameters[param_name][~remove_mask]
        self.K -= 1
    
    def remove_final_parameter(self):
        remove_mask = torch.tensor([(i+1)==self.K for i in range(self.K)])
        self.remove_parameters(remove_mask)

    def update_learnable_parameters(self, loss, lr):
        for param_name, is_learnable in self.learnable_parameters.items():
            if is_learnable:
                # print(f"updating: {param_name}: {self.parameters[param_name]}")
                grad = torch.autograd.grad(loss, self.parameters[param_name], retain_graph=True)[0]
                self.parameters[param_name] = self.parameters[param_name] + lr * grad
                self.parameters[param_name] = self.parameters[param_name].detach().requires_grad_(True)
                # print(f"After update: {param_name}: {self.parameters[param_name]}")
            # else:
            #     print(f"not updating: {param_name}")