import numpy as np

def markov(rho, A, nmax, rng):

    ####################### Tests
    A_shape = A.shape
    assert A_shape[0] == A_shape[1]
    N = A_shape[0]
    assert N == rho.shape[0]
    assert np.isclose(np.sum(rho), 1)
    assert np.allclose(np.sum(A, axis=1), 1)
    #########################

    states = np.arange(N)

    X_list = np.zeros(nmax + 1, dtype=int)
    X_list[0] = rng.choice(states, p = rho.flatten())

    for q in range(1, nmax + 1):
        current_state = X_list[q-1]
        X_list[q] = np.random.choice(states, p = A[current_state])

    return X_list
