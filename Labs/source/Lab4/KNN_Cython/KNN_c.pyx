import numpy as np
import bottleneck as bn
from libc.math cimport sqrt
cimport cython

DTYPE = np.double

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef long[:] KNN_c(double[:,:] x_train, double[:] class_train, double[:,:] x_test, Py_ssize_t k):

    distances = np.zeros((x_test.shape[0],x_train.shape[0]), dtype=DTYPE)
    cdef double[:,:] distances_view = distances
    cdef Py_ssize_t i,j
    for i in range(x_train.shape[0]):
        for j in range(x_test.shape[0]):
            distances_view[j,i]= sqrt((x_train[i,0]-x_test[j,0])*(x_train[i,0]-x_test[j,0]) +(x_train[i,1]-x_test[j,1])*(x_train[i,1]-x_test[j,1]) )

    id = bn.argpartition(distances_view, k, axis=1)[:, :k]  # On récupère les indices des k plus petites distances
    labels = np.array(class_train)[id.astype(int)]
    
    class_pred = np.zeros(x_test.shape[0], dtype=int)
    cdef long[:] class_pred_view = class_pred

    for i in range(x_test.shape[0]):
        class_pred_view[i] = np.bincount(labels[i].astype(np.int64)).argmax()

    return class_pred_view