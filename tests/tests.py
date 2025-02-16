import pytest
import torch
import algorithms.sva
from algorithms.sva import calculate_cluster_similarity

def test_calculate_cluster_similarity():
    K = 3

    rho1 = torch.tensor([[
        [0, 0, 1.],
        [0, 0, 1.],
        [0, 0, 1.]
    ]])
    rho2 = torch.tensor([[
        [0, 0, 1.],
        [0, 0, 1.],
        [0, 0, 1.]
    ]])
    rho3 = torch.tensor([[
        [1., 0, 0],
        [1., 0, 0],
        [1., 0, 0]
    ]])
    rho = torch.cat((rho1, rho2, rho3))

    actual = calculate_cluster_similarity(rho, K)
    expected = torch.tensor(
        [
            [0, 0, 2/3],
            [0, 0, 2/3],
            [0, 0, 0]
        ]
    )

    assert (actual==expected).all()

    # def calculate_cluster_similarity(rho, K):
    # ## TODO: make similarity metric an arg?
    # return (
    #     torch.tensor(
    #         [l1_loss(rho[:,i], rho[:,j], reduction='mean') if j > i else 0. 
    #         for i in range(K) for j in range(K)]
    #     )
    #     .view(K, K)
    # )


