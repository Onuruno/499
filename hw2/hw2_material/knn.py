import numpy as np
import matplotlib.pyplot as plt


def calculate_distances(train_data, test_instance, distance_metric):
    """
    Calculates Manhattan (L1) / Euclidean (L2) distances between test_instance and every train instance.
    :param train_data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data.
    :param test_instance: A (D, ) shaped numpy array.
    :param distance_metric: A string which indicates the distance metric, it can be either 'L1' or 'L2'
    :return: An (N, ) shaped numpy array that contains distances.
    """
    if(distance_metric == 'L1'):
        distances = np.sum(np.abs(train_data-test_instance), axis=1)
        return distances
    elif(distance_metric == 'L2'):
        distances = np.sqrt(np.sum(np.power((train_data-test_instance), 2), axis=1))
        return distances

def majority_voting(distances, labels, k):
    """
    Applies majority voting. If there are more then one major class, returns the smallest label.
    :param distances: An (N, ) shaped numpy array that contains distances
    :param labels: An (N, ) shaped numpy array that contains labels
    :param k: An integer. The number of nearest neighbor to be selected.
    :return: An integer. The label of the majority class.
    """
    indices = distances.argsort()[:k]
    votes = labels[indices]
    counts = np.bincount(votes)
    return np.argmax(counts)


def knn(train_data, train_labels, test_data, test_labels, k, distance_metric):
    """
    Calculates accuracy of knn on test data using train_data.
    :param train_data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param train_labels: An (N, ) shaped numpy array that contains labels
    :param test_data: An (M, D) shaped numpy array where M is the number of examples
    and D is the dimension of the data
    :param test_labels: An (M, ) shaped numpy array that contains labels
    :param k: An integer. The number of nearest neighbor to be selected.
    :param distance_metric: A string which indicates the distance metric, it can be either 'L1' or 'L2'
    :return: A float. The calculated accuracy.
    """
    predicted_labels = [majority_voting(calculate_distances(train_data, test_instance, distance_metric), train_labels, k) for test_instance in test_data]
    
    return np.count_nonzero((predicted_labels-test_labels) == 0)/len(predicted_labels)


def split_train_and_validation(whole_train_data, whole_train_labels, validation_index, k_fold):
    """
    Splits training dataset into k and returns the validation_indexth one as the
    validation set and others as the training set. You can assume k_fold divides N.
    :param whole_train_data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param whole_train_labels: An (N, ) shaped numpy array that contains labels
    :param validation_index: An integer. 0 <= validation_index < k_fold. Specifies which fold
    will be assigned as validation set.
    :param k_fold: The number of groups that the whole_train_data will be divided into.
    :return: train_data, train_labels, validation_data, validation_labels
    train_data.shape is (N-N/k_fold, D).
    train_labels.shape is (N-N/k_fold, ).
    validation_data.shape is (N/k_fold, D).
    validation_labels.shape is (N/k_fold, ).
    """
    n = len(whole_train_data)
    size = n//k_fold
    train_data = np.concatenate((whole_train_data[:validation_index*size], whole_train_data[(validation_index+1)*size:]))
    train_labels = np.concatenate((whole_train_labels[:validation_index*size], whole_train_labels[(validation_index+1)*size:]))
    validation_data = whole_train_data[validation_index*size:(validation_index+1)*size]
    validation_labels = whole_train_labels[validation_index*size:(validation_index+1)*size]
    
    return train_data, train_labels, validation_data, validation_labels 

def cross_validation(whole_train_data, whole_train_labels, k_fold, k, distance_metric):
    """
    Applies k_fold cross-validation and averages the calculated accuracies.
    :param whole_train_data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param whole_train_labels: An (N, ) shaped numpy array that contains labels
    :param k_fold: An integer.
    :param k: An integer. The number of nearest neighbor to be selected.
    :param distance_metric: A string which indicates the distance metric, it can be either 'L1' or 'L2'
    :return: A float. Average accuracy calculated.
    """
    accuracy = 0.
    for i in range(k_fold):
        train_data, train_labels, validation_data, validation_labels = split_train_and_validation(whole_train_data, whole_train_labels, i, k_fold)
        accuracy += knn(train_data, train_labels, validation_data, validation_labels, k, distance_metric)
    accuracy /= k_fold
    return accuracy

train_set = np.load('knn/train_set.npy')
train_labels = np.load('knn/train_labels.npy')
test_set = np.load('knn/test_set.npy')
test_labels = np.load('knn/test_labels.npy')

accuracies_L1 = np.array([cross_validation(train_set, train_labels, 10, i, 'L1') for i in range(1,180)])
suitible_k_L1 = np.argmax(accuracies_L1)+1

plt.subplot(1,2,1)
plt.plot(np.arange(1,180), accuracies_L1)
plt.xlabel('k value')
plt.ylabel('Accuracy')
plt.title('L1 accuracies')

accuracies_L2 = np.array([cross_validation(train_set, train_labels, 10, i, 'L2') for i in range(1,180)])
suitible_k_L2 = np.argmax(accuracies_L2)+1

plt.subplot(1,2,2)
plt.plot(np.arange(1,180), accuracies_L2)
plt.xlabel('k value')
plt.title('L2 accuracies')

fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
fig.savefig('knn_accuracies.png', dpi=100)

test_accuracy_L1 = knn(train_set, train_labels, test_set, test_labels, suitible_k_L1, 'L1')
test_accuracy_L2 = knn(train_set, train_labels, test_set, test_labels, suitible_k_L2, 'L2')
print('Test accuracy for L1: ' + str(test_accuracy_L1))
print('Test accuracy for L2: ' + str(test_accuracy_L2))