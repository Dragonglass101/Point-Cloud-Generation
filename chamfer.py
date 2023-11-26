import numpy as np
from scipy.spatial import KDTree

def chamfer_distance(A, B):
    """
    Computes the chamfer distance between two sets of points A and B.
    """
    tree = KDTree(B)
    dist_A = tree.query(A)[0]
    tree = KDTree(A)
    dist_B = tree.query(B)[0]
    return np.mean(dist_A) + np.mean(dist_B)


number_points = 1000
pcd1 = np.random.rand(number_points, 3)  # uniform distribution over [0, 1)
pcd2 = np.random.rand(number_points, 3)  # uniform distribution over [0, 1)


print(chamfer_distance(pcd1, pcd2))