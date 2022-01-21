import numpy as np

def forward(A, B, pi, O):
    """
    Calculates the probability of an observation sequence O given the model(A, B, pi).
    :param A: state transition probabilities (NxN)
    :param B: observation probabilites (NxM)
    :param pi: initial state probabilities (N)
    :param O: sequence of observations(T) where observations are just indices for the columns of B (0-indexed)
        N is the number of states,
        M is the number of possible observations, and
        T is the sequence length.
    :return: The probability of the observation sequence and the calculated alphas in the Trellis diagram with shape
             (N, T) which should be a numpy array.
    """
    N = len(A)
    M = len(B[0])
    T = len(O)
    
    result = np.zeros((N, T))
    
    for j in range(N):
        result[j][0] = pi[j]*B[j][O[0]]
    
    for j in range(1,T):
        for i in range(N):
            for k in range(N):
                result[i][j] += result[k][j-1]*A[k][i]*B[i][O[j]]
    return np.sum(result, axis=0)[T-1], result  

def viterbi(A, B, pi, O):
    """
    Calculates the most likely state sequence given model(A, B, pi) and observation sequence.
    :param A: state transition probabilities (NxN)
    :param B: observation probabilites (NxM)
    :param pi: initial state probabilities(N)
    :param O: sequence of observations(T) where observations are just indices for the columns of B (0-indexed)
        N is the number of states,
        M is the number of possible observations, and
        T is the sequence length.
    :return: The most likely state sequence with shape (T,) and the calculated deltas in the Trellis diagram with shape
             (N, T). They should be numpy arrays.
    """
    N = len(A)
    M = len(B[0])
    T = len(O)
    
    result = np.zeros((N, T))
    
    for j in range(N):
        result[j][0] = pi[j]*B[j][O[0]]
    
    for j in range(1,T):
        for i in range(N):
            values = np.zeros(N)
            for k in range(N):
                values[k] = result[k][j-1]*A[k][i]*B[i][O[j]]
            result[i][j] = np.max(values)
    max_values = np.argmax(result, axis=0)
    return max_values, result