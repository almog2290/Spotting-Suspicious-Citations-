import numpy as np
from algos import floyd_warshall

adjacency_matrix = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]])
M, path = floyd_warshall(adjacency_matrix)

print(M)