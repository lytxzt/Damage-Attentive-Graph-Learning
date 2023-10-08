import numpy as np
import Utils
import random

positions= np.array([[random.randint(0, 100), random.randint(0, 100)] for _ in range(100)])
# print(positions)
A = Utils.make_A_matrix(positions, len(positions), 100)
D = Utils.make_D_matrix(A, len(positions))
L = D - A
connected_flag, num_of_clusters = Utils.check_number_of_clusters(L, len(positions))
# print(num_of_clusters)

# e_vals, e_vecs = np.linalg.eig(A)
# # print(e_vals, e_vecs)
# print(max(e_vals), np.argmax(e_vals))
# print(e_vecs[:, np.argmax(e_vals)], np.var(e_vecs[np.argmax(e_vals),:]))

degree = np.sum(A, axis=0)
print(degree, np.var(degree))
