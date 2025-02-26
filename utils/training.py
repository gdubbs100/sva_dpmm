import torch
import torch.distributions as dist
from torch.nn.functional import l1_loss


def calculate_rho(data, w, phi, sigma):
    ## TODO: figure out how to make this distribution more generic
    h=torch.vstack([dist.MultivariateNormal(_phi, sigma).log_prob(data) for _phi in phi])
    log_weights = torch.log(w) + h.T # Compute log of weights * likelihood
    log_rho = log_weights - torch.logsumexp(log_weights, dim=1, keepdim=True)  # Normalize
    return torch.exp(log_rho).squeeze()

# 2. calculate cluster similarity
def calculate_cluster_distance(rho, K):
    ## TODO: make similarity metric an arg?
    return (
        torch.tensor(
            [l1_loss(rho[:,i], rho[:,j], reduction='mean') if j > i else 0. 
            for i in range(K) for j in range(K)]
        )
        .view(K, K)
    )

# 3. identify those clusters to merge
def get_clusters_to_merge(cluster_distance_matrix, distance_threshold):
    mask = (cluster_distance_matrix > 0) & (cluster_distance_matrix < distance_threshold)
    to_merge = torch.nonzero(mask, as_tuple=False)
    values_to_merge = cluster_distance_matrix[mask]
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

def merge_and_prune(data, w, phi, rho, sigma, K, distance_threshold):
    rho_prime = calculate_rho(data, w, phi, sigma)
    cluster_distance_matrix = calculate_cluster_distance(rho_prime, K)
    clusters_to_merge = get_clusters_to_merge(cluster_distance_matrix, distance_threshold)
    merge_statistics, all_to_prune = get_merge_statistics(clusters_to_merge, phi, w, K)
    phi, w = apply_merge(merge_statistics, phi, w)
    w, phi, K = prune(all_to_prune, w, phi)
    return w, phi, K