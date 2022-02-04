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

def getMax(arr):            #returns maximum element of an array
    value = arr[0][1]
    idx = arr[0][0]
    for item in arr[1:]:
        if(item[1] > value):
            value = item[1]
            idx = item[0]
    return idx, value

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
    
    result = []
    for i in range(N):
        arr = []
        for j in range(T):
            arr.append((None, 0.0))
        result.append(arr)
    
    for j in range(N):
        result[j][0] = (None, pi[j]*B[j][O[0]])
    
    for j in range(1,T):
        for i in range(N):
            values = []
            for k in range(N):
                values.append((k, result[k][j-1][1]*A[k][i]*B[i][O[j]]))
            result[i][j] = getMax(values)

    state_sequence = []
    idx, final_prob = getMax([row[-1] for row in result])
    state_sequence.insert(0, [row[-1] for row in result].index((idx, final_prob)))
    state_sequence.insert(0, idx)
    
    for m in range(T-2, 0, -1):
        idx = result[idx][m][0]
        state_sequence.insert(0, idx)
    
    state_sequence = np.array(state_sequence)
    
    delta = np.zeros((0,T))
    
    for r in range(N):
        delta = np.append(delta, [[element[1] for element in result[r]]], axis=0)
    
    return state_sequence, delta
