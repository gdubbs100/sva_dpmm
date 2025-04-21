import torch
import torch.distributions as dist
from torch.nn.functional import l1_loss


def calculate_rho(data, _w, mixture_dist):
    ## unsqueeze at -2 lets first dim be interpreted as batch size
    _h = mixture_dist.log_prob(data.unsqueeze(-2))
    log_weights = torch.log(_w) + _h # Compute log of weights * likelihood

    log_rho = log_weights - torch.logsumexp(log_weights, dim=-1, keepdim=True)  # Normalize

    return torch.exp(log_rho)

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
def get_merge_mapping(clusters_to_merge, K):
    merged = []
    all_to_prune = []
    merge_statistics = dict()

    for m in clusters_to_merge:
        if (m[0] not in merged) and (m[1] not in merged):
            all_to_prune.append(m[0].item()) # prune first idx
            # map first cluster to second
            merge_statistics[m[0].item()] = m[1].item()
            merged.append(m[0])
            merged.append(m[1])
    
    # convert to boolean tensor
    all_to_prune = torch.tensor([i in all_to_prune for i in range(K)])
    return merge_statistics, all_to_prune

## apply merge, then apply prune
def merge_and_prune_weights(merge_statistics, w):
    mask = torch.zeros(w.size(0))!=0
    ## merge
    for k, v in merge_statistics.items():
        w[v] = w[k] + w[v]
        mask[k] = True # don't prune value
    ## prune
    w = prune_weights(w, mask)
    return w

def prune_weights(w, mask):
    w = w[~mask]
    return w

def prune(prune_idx: torch.Tensor, w: torch.Tensor, phi: torch.Tensor):
    w = w[~prune_idx]
    phi = phi[~prune_idx]
    K = w.size(0)
    return w, phi, K

def get_merge_statistics(data, w, mixture_dist, distance_threshold):
    rho_prime = calculate_rho(data, w, mixture_dist)
    cluster_distance_matrix = calculate_cluster_distance(rho_prime, mixture_dist.K)
    clusters_to_merge = get_clusters_to_merge(cluster_distance_matrix, distance_threshold)
    return get_merge_mapping(clusters_to_merge, mixture_dist.K)
