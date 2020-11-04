import numpy as np
from gfs import PygfsManager


def test0():
    feat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 1, 1]])
    labels = [1, 3, 4, 5]

    gfsMan = PygfsManager(feat, labels, "shannon".encode())
    print("Selected features ---> ", gfsMan.greedyAlgorithm(2))
    gfsMan = PygfsManager(feat, labels, "renyi".encode())
    print("Selected features ---> ", gfsMan.greedyAlgorithm(2))



if __name__ == '__main__':
    test0()
