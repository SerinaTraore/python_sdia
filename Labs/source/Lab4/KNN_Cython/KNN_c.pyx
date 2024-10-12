import numpy as np
import bottleneck as bn
from libc.math cimport sqrt
cimport cython

DTYPE = np.double

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef KNN_c(double[:,:] x_train, double[:] class_train, double[:,:] x_test, Py_ssize_t k):

    distances = np.zeros((x_test.shape[0],x_train.shape[0]), dtype=DTYPE)
    cdef double[:,:] distances_view = distances
    cdef Py_ssize_t i,j
    for i in range(x_train.shape[0]):
        for j in range(x_test.shape[0]):
            distances_view[j,i]= sqrt((x_train[i,0]-x_test[j,0])*(x_train[i,0]-x_test[j,0]) +(x_train[i,1]-x_test[j,1])*(x_train[i,1]-x_test[j,1]) )

    id = bn.argpartition(distances, k, axis=1)[:, :k]  # On récupère les indices des k plus petites distances
    closest_k_distances = np.take_along_axis(distances, id, axis=1)  # On trie ces k distances
    id_sorted = np.argsort(closest_k_distances, axis=1)  # Trie les indices par ordre croissant de distance
    final_ids = np.take_along_axis(id, id_sorted, axis=1).astype(np.int64)

    labels = np.array(class_train)[final_ids.astype(int)]
    
    class_pred = np.zeros(x_test.shape[0], dtype=int)

    for i in range(x_test.shape[0]):
        class_pred[i] = np.bincount(labels[i].astype(np.int64)).argmax()

    return class_pred