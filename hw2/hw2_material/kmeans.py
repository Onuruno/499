import numpy as np
import matplotlib.pyplot as plt

def findClosestClusterCenter(point, cluster_centers):
    distances = np.sum(np.power((cluster_centers-point), 2), axis=1)
    return np.argmin(distances)

def calculate_objective_function(data, assignments, cluster_centers, k):
    temp_cluster_centers = np.zeros(data.shape)
    for i in range(len(data)):
        temp_cluster_centers[i] = cluster_centers[int(assignments[i])]
    return np.sum(np.sum(np.power((data-temp_cluster_centers), 2), axis=1))*0.5
        
def assign_clusters(data, cluster_centers):
    """
    Assigns every data point to its closest (in terms of Euclidean distance) cluster center.
    :param data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param cluster_centers: A (K, D) shaped numpy array where K is the number of clusters
    and D is the dimension of the data
    :return: An (N, ) shaped numpy array. At its index i, the index of the closest center
    resides to the ith data point.
    """

    assignments = np.array([findClosestClusterCenter(single_data, cluster_centers) for single_data in data])
    return assignments

def calculate_cluster_centers(data, assignments, cluster_centers, k):
    """
    Calculates cluster_centers such that their squared Euclidean distance to the data assigned to
    them will be lowest.
    If none of the data points belongs to some cluster center, then assign it to its previous value.
    :param data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param assignments: An (N, ) shaped numpy array with integers inside. They represent the cluster index
    every data assigned to.
    :param cluster_centers: A (K, D) shaped numpy array where K is the number of clusters
    and D is the dimension of the data
    :param k: Number of clusters
    :return: A (K, D) shaped numpy array that contains the newly calculated cluster centers.
    """
    new_cluster_centers = np.zeros(cluster_centers.shape)
    counters = np.zeros(k)
    
    for i in range(len(data)):
        new_cluster_centers[int(assignments[i])] += data[i]
        counters[int(assignments[i])] +=1
    for i in range(k):
        if(counters[i] == 0):
            new_cluster_centers[i] = cluster_centers[i]
            counters[i] = 1
    new_cluster_centers /= (counters.reshape(-1,1))
    return new_cluster_centers

def kmeans(data, initial_cluster_centers):
    """
    Applies k-means algorithm.
    :param data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param initial_cluster_centers: A (K, D) shaped numpy array where K is the number of clusters
    and D is the dimension of the data
    :return: cluster_centers, objective_function
    cluster_center.shape is (K, D).
    objective function is a float. It is calculated by summing the squared euclidean distance between
    data points and their cluster centers.
    """
    
    k = initial_cluster_centers.shape[0]
    assignments = np.zeros(data.shape[0])
    cluster_centers = np.copy(initial_cluster_centers)
    objective_function = 0.0
    
    while(True):
        assignments = assign_clusters(data, cluster_centers)
        cluster_centers = calculate_cluster_centers(data, assignments, cluster_centers, k)
        temp_objective_function = calculate_objective_function(data, assignments, cluster_centers, k)
        if(temp_objective_function == objective_function):
            break
        objective_function = temp_objective_function
    return cluster_centers, objective_function

for l in range(10):
    fig, axs = plt.subplots(2, 2)
    
    dataset1 = np.load('kmeans/dataset1.npy')
    objective_function_values1 = np.zeros(11)
    for k in range(1,11):
        index = np.random.randint(0, dataset1.shape[0], size=k)
        initial_cluster_centers = np.take(dataset1, index, axis=0)
        cluster_centers, objective_function = kmeans(dataset1, initial_cluster_centers)
        objective_function_values1[k] = objective_function
    axs[0, 0].plot(np.arange(11), objective_function_values1)
    axs[0, 0].set_title('Dataset1')
    
    dataset2 = np.load('kmeans/dataset2.npy')
    objective_function_values2 = np.zeros(11)
    for k in range(1,11):
        index = np.random.randint(0, dataset2.shape[0], size=k)
        initial_cluster_centers = np.take(dataset2, index, axis=0)
        cluster_centers, objective_function = kmeans(dataset2, initial_cluster_centers)
        objective_function_values2[k] = objective_function
    axs[0, 1].plot(np.arange(11), objective_function_values2)
    axs[0, 1].set_title('Dataset2')

    dataset3 = np.load('kmeans/dataset3.npy')
    objective_function_values3 = np.zeros(11)
    for k in range(1,11):
        index = np.random.randint(0, dataset3.shape[0], size=k)
        initial_cluster_centers = np.take(dataset3, index, axis=0)
        cluster_centers, objective_function = kmeans(dataset3, initial_cluster_centers)
        objective_function_values3[k] = objective_function
    axs[1, 0].plot(np.arange(11), objective_function_values3)
    axs[1, 0].set_title('Dataset3')

    dataset4 = np.load('kmeans/dataset4.npy')
    objective_function_values4 = np.zeros(11)
    for k in range(1,11):
        index = np.random.randint(0, dataset4.shape[0], size=k)
        initial_cluster_centers = np.take(dataset4, index, axis=0)
        cluster_centers, objective_function = kmeans(dataset4, initial_cluster_centers)
        objective_function_values4[k] = objective_function
    axs[1, 1].plot(np.arange(11), objective_function_values4)
    axs[1, 1].set_title('Dataset4')

    for ax in axs.flat:
        ax.set(xlabel='k', ylabel='Objective_function')
    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()
        
    plt.savefig('plot'+str(l+1)+'.png')

colors = ['red', 'green', 'blue', 'yellow']

fig, axs = plt.subplots(2, 2)

dataset1 = np.load('kmeans/dataset1.npy')
index = np.random.randint(0, dataset1.shape[0], size=2)
initial_cluster_centers = np.take(dataset1, index, axis=0)
cluster_centers, objective_function = kmeans(dataset1, initial_cluster_centers)
assignments = assign_clusters(dataset1, cluster_centers)
for i in range(len(assignments)):
    axs[0,0].scatter(dataset1[i][0], dataset1[i][1], s=1, color=colors[int(assignments[i])])
for j in range(2):
    axs[0,0].plot(cluster_centers[j][0], cluster_centers[j][1], color='black', marker='^')
axs[0,0].set_title('Dataset1')
        
dataset2 = np.load('kmeans/dataset2.npy')
index = np.random.randint(0, dataset2.shape[0], size=3)
initial_cluster_centers = np.take(dataset2, index, axis=0)
cluster_centers, objective_function = kmeans(dataset2, initial_cluster_centers)
assignments = assign_clusters(dataset2, cluster_centers)
for i in range(len(assignments)):
    axs[0,1].scatter(dataset2[i][0], dataset2[i][1], s=1, color=colors[int(assignments[i])])
for j in range(3):
    axs[0,1].plot(cluster_centers[j][0], cluster_centers[j][1], color='black', marker='^')
axs[0,1].set_title('Dataset2')
        
dataset3 = np.load('kmeans/dataset3.npy')
index = np.random.randint(0, dataset3.shape[0], size=4)
initial_cluster_centers = np.take(dataset3, index, axis=0)
cluster_centers, objective_function = kmeans(dataset3, initial_cluster_centers)
assignments = assign_clusters(dataset3, cluster_centers)
for i in range(len(assignments)):
    axs[1,0].scatter(dataset3[i][0], dataset3[i][1], s=1, color=colors[int(assignments[i])])
for j in range(4):
    axs[1,0].plot(cluster_centers[j][0], cluster_centers[j][1], color='black', marker='^')
axs[1,0].set_title('Dataset3')
        
dataset4 = np.load('kmeans/dataset4.npy')
index = np.random.randint(0, dataset4.shape[0], size=4)
initial_cluster_centers = np.take(dataset4, index, axis=0)
cluster_centers, objective_function = kmeans(dataset4, initial_cluster_centers)
assignments = assign_clusters(dataset4, cluster_centers)
for i in range(len(assignments)):
    axs[1,1].scatter(dataset4[i][0], dataset4[i][1], s=1, color=colors[int(assignments[i])])
for j in range(4):
    axs[1,1].plot(cluster_centers[j][0], cluster_centers[j][1], color='black', marker='^')
axs[1,1].set_title('Dataset4')

plt.savefig('Clustered_Data.png')