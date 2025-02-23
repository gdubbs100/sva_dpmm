import pytest
import torch
import algorithms.sva
from algorithms.sva import calculate_cluster_similarity, prune

def test_calculate_cluster_similarity():
    K = 3

    rho = torch.tensor(
        [
            [0, 0, 1.],
            [1., 0, 0],
            [0, 0, 1.]
        ],
    )


    actual = calculate_cluster_similarity(rho, K)
    expected = torch.tensor(
        [
            [0, 2/3, 0],
            [0, 0, 2/3],
            [0, 0, 0]
        ]
    )

    assert (actual==expected).all()
    
def test_prune():
    w = torch.tensor(
        [.1, .1, .2, .6]
    )
    phi = torch.tensor(
        [
            [1,1],
            [2,2],
            [3,3],
            [4,4]
        ]
    )
    K = phi.size(0)
    to_prune = w < .2

    w, phi, K = prune(to_prune, w, phi)

    assert K == 2
    assert (w == torch.tensor([.2, .6])).all()
    assert (phi == torch.tensor([[3,3],[4,4]])).all()


