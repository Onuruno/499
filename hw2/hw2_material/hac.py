import numpy as np


def single_linkage(c1, c2):
    """
    Given clusters c1 and c2, calculates the single linkage criterion.
    :param c1: An (N, D) shaped numpy array containing the data points in cluster c1.
    :param c2: An (M, D) shaped numpy array containing the data points in cluster c2.
    :return: A float. The result of the calculation.
    """
    return np.amin([np.amin(np.sqrt(np.sum(np.power((c1-item), 2), axis=1))) for item in c2])

def complete_linkage(c1, c2):
    """
    Given clusters c1 and c2, calculates the complete linkage criterion.
    :param c1: An (N, D) shaped numpy array containing the data points in cluster c1.
    :param c2: An (M, D) shaped numpy array containing the data points in cluster c2.
    :return: A float. The result of the calculation.
    """
    return np.amax([np.amax(np.sqrt(np.sum(np.power((c1-item), 2), axis=1))) for item in c2])

def average_linkage(c1, c2):
    """
    Given clusters c1 and c2, calculates the average linkage criterion.
    :param c1: An (N, D) shaped numpy array containing the data points in cluster c1.
    :param c2: An (M, D) shaped numpy array containing the data points in cluster c2.
    :return: A float. The result of the calculation.
    """
    n = len(c1)
    m = len(c2)
    return np.sum([(np.sum(np.sqrt(np.sum(np.power((c1-item), 2), axis=1)))/m) for item in c2])/n

def centroid_linkage(c1, c2):
    """
    Given clusters c1 and c2, calculates the centroid linkage criterion.
    :param c1: An (N, D) shaped numpy array containing the data points in cluster c1.
    :param c2: An (M, D) shaped numpy array containing the data points in cluster c2.
    :return: A float. The result of the calculation.
    """
    return np.sqrt(np.sum(np.power((np.sum(c1, axis=0)/len(c1)-np.sum(c2, axis=0)/len(c2)), 2)))
    


def hac(data, criterion, stop_length):
    """
    Applies hierarchical agglomerative clustering algorithm with the given criterion on the data
    until the number of clusters reaches the stop_length.
    :param data: An (N, D) shaped numpy array containing all of the data points.
    :param criterion: A function. It can be single_linkage, complete_linkage, average_linkage, or
    centroid_linkage
    :param stop_length: An integer. The length at which the algorithm stops.
    :return: A list of numpy arrays with length stop_length. Each item in the list is a cluster
    and a (Ni, D) sized numpy array.
    """
    result_list = [[item] for item in data]
    while(len(result_list) > stop_length):
        n = len(result_list)
        smallest_distance = criterion(result_list[0], result_list[1])
        c1, c2 = 0, 1
        for i in range(n):
            for j in range(i+1, n):
                temp_value = criterion(result_list[i], result_list[j])
                if(temp_value < smallest_distance):
                    smallest_distance = temp_value
                    c1, c2 = i, j
        result_list[c1] = np.concatenate((result_list[c1], result_list[c2]), axis=0)
        del result_list[c2]
    return result_list