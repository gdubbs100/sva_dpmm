import pytest
import torch
import algorithms.sva
from utils.training import calculate_cluster_distance #, prune
from utils.lr_scheduler import SVALRScheduler

def test_calculate_cluster_distance():
    K = 3

    rho = torch.tensor(
        [
            [0, 0, 1.],
            [1., 0, 0],
            [0, 0, 1.]
        ],
    )


    actual = calculate_cluster_distance(rho, K)
    expected = torch.tensor(
        [
            [0, 1/3, 1],
            [0, 0, 2/3],
            [0, 0, 0]
        ]
    )

    assert (actual==expected).all()
    
# def test_prune():
#     w = torch.tensor(
#         [.1, .1, .2, .6]
#     )
#     phi = torch.tensor(
#         [
#             [1,1],
#             [2,2],
#             [3,3],
#             [4,4]
#         ]
#     )
#     K = phi.size(0)
#     to_prune = w < .2

#     w, phi, K = prune(to_prune, w, phi)

#     assert K == 2
#     assert (w == torch.tensor([.2, .6])).all()
#     assert (phi == torch.tensor([[3,3],[4,4]])).all()

def test_sva_lr_scheduler():
    lr=1
    gamma = .6
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = SVALRScheduler(optimizer, gamma=gamma)
    for i in range(1, 10):
        optimizer.step()
        scheduler.step()
        assert scheduler.get_last_lr()[0] == 1/(i + 1)**gamma
