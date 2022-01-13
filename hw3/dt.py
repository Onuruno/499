import numpy as np
def entropy(bucket):
    """
    Calculates the entropy.
    :param bucket: A list of size num_classes. bucket[i] is the number of
    examples that belong to class i.
    :return: A float. Calculated entropy.
    """
    total_example = 0
    for item in bucket:
        total_example += item
    entropy = 0
    if(total_example == 0):
        return entropy
    for i in range(len(bucket)):
        p = (bucket[i]/total_example)
        if(p == 0):
            continue
        entropy -= p*np.log2(p)
    return entropy


def info_gain(parent_bucket, left_bucket, right_bucket):
    """
    Calculates the information gain. A bucket is a list of size num_classes.
    bucket[i] is the number of examples that belong to class i.
    :param parent_bucket: Bucket belonging to the parent node. It contains the
    number of examples that belong to each class before the split.
    :param left_bucket: Bucket belonging to the left child after the split.
    :param right_bucket: Bucket belonging to the right child after the split.
    :return: A float. Calculated information gain.
    """
    left_example = 0
    for item in left_bucket:
        left_example += item
    right_example = 0
    for item in right_bucket:
        right_example += item
    average_entropy = (left_example/(left_example+right_example)*entropy(left_bucket)) + (right_example/(left_example+right_example)*entropy(right_bucket))
    return entropy(parent_bucket) - average_entropy

def gini(bucket):
    """
    Calculates the gini index.
    :param bucket: A list of size num_classes. bucket[i] is the number of
    examples that belong to class i.
    :return: A float. Calculated gini index.
    """
    total_example = 0
    for item in bucket:
        total_example += item
    gini = 1
    for item in bucket:
        gini -= np.power((item/total_example), 2)
    return gini

def avg_gini_index(left_bucket, right_bucket):
    """
    Calculates the average gini index. A bucket is a list of size num_classes.
    bucket[i] is the number of examples that belong to class i.
    :param left_bucket: Bucket belonging to the left child after the split.
    :param right_bucket: Bucket belonging to the right child after the split.
    :return: A float. Calculated average gini index.
    """
    left_example = 0
    for item in left_bucket:
        left_example += item
    right_example = 0
    for item in right_bucket:
        right_example += item
    return (left_example/(left_example+right_example)*gini(left_bucket)) + (right_example/(left_example+right_example)*gini(right_bucket))

def calculate_split_values(data, labels, num_classes, attr_index, heuristic_name):
    """
    For every possible values to split the data for the attribute indexed by
    attribute_index, it divides the data into buckets and calculates the values
    returned by the heuristic function named heuristic_name. The split values
    should be the average of the closest 2 values. For example, if the data has
    2.1 and 2.2 in it consecutively for the values of attribute index by attr_index,
    then one of the split values should be 2.15.
    :param data: An (N, M) shaped numpy array. N is the number of examples in the
    current node. M is the dimensionality of the data. It contains the values for
    every attribute for every example.
    :param labels: An (N, ) shaped numpy array. It contains the class values in
    it. For every value, 0 <= value < num_classes.
    :param num_classes: An integer. The number of classes in the dataset.
    :param attr_index: An integer. The index of the attribute that is going to
    be used for the splitting operation. This integer indexs the second dimension
    of the data numpy array.
    :param heuristic_name: The name of the heuristic function. It should either be
    'info_gain' of 'avg_gini_index' for this homework.
    :return: An (L, 2) shaped numpy array. L is the number of split values. The
    first column is the split values and the second column contains the calculated
    heuristic values for their splits.
    """
    values = np.zeros((data.shape[0]-1, 2))
    data_column = data[:,attr_index]
    sorted_data = np.sort(data_column)
    
    values[:,0] = (sorted_data[1:]+sorted_data[:-1])/2
        
    parent_bucket = np.bincount(labels, minlength=num_classes)
    for i in range(values.shape[0]):
        left_bucket = np.zeros(num_classes, dtype=int)
        right_bucket = np.zeros(num_classes, dtype=int)
        for item in range(len(data_column)):
            if(data_column[item] < values[i][0]):
                left_bucket[labels[item]] +=1
            else:
                right_bucket[labels[item]] +=1
        if(heuristic_name == 'info_gain'):
            values[i][1] = info_gain(parent_bucket, left_bucket, right_bucket)
        else:
            values[i][1] = avg_gini_index(left_bucket, right_bucket)
    return values  

def chi_squared_test(left_bucket, right_bucket):
    """
    Calculates chi squared value and degree of freedom between the selected attribute
    and the class attribute. A bucket is a list of size num_classes. bucket[i] is the
    number of examples that belong to class i.
    :param left_bucket: Bucket belonging to the left child after the split.
    :param right_bucket: Bucket belonging to the right child after the split.
    :return: A float and and integer. Chi squared value and degree of freedom.
    """
    bucket = np.array([left_bucket, right_bucket])
    bucket = bucket[:,~np.all(bucket == 0, axis=0)]

    (row, column) = bucket.shape
    
    expected = np.zeros((row, column))
    for i in range(row):
        for j in range(column):
            expected[i][j] = (np.sum(bucket,axis=0)[j]*np.sum(bucket, axis=1)[i])/np.sum(bucket)
    difference = bucket-expected
    expected[expected == 0] = 1
    chi_squared = np.sum(np.power(difference, 2)/expected)
    dof = (column-1) * (row-1)
    return chi_squared, dof

def bestSplit(data, labels, num_classes, heuristic_name):
    num_attr = data.shape[1]
    
    best_values = np.zeros((num_attr,2))
    for attr_index in range(num_attr):
        values = calculate_split_values(data, labels, num_classes, attr_index, heuristic_name)
        idx = np.argmax(values[:,1])
        best_values[attr_index] = values[idx]
    best_attr_index = np.argmax(best_values[:,1])
    split_value = best_values[best_attr_index][0]
    return split_value, best_attr_index
    
def splitData(data, labels, split_value, best_attr_index):
    left_data = np.zeros((0, data.shape[1]))
    right_data = np.zeros((0, data.shape[1]))
    left_labels = np.zeros((1,0), dtype=int)
    right_labels = np.zeros((1,0), dtype=int)
    
    for i in range(data.shape[0]):
        if(data[i][best_attr_index] < split_value):
            left_data = np.vstack((left_data, data[i]))
            left_labels = np.append(left_labels, labels[i])
        else:
            right_data = np.vstack((right_data, data[i]))
            right_labels = np.append(right_labels, labels[i])
    return left_data, left_labels, right_data, right_labels

def getBucket(labels, num_classes):
    return np.bincount(labels, minlength=num_classes)

def shouldSplit(left_bucket, right_bucket):
    critical_values = [0.0157,0.211]
    chi_squared, dof = chi_squared_test(left_bucket, right_bucket)
    if(chi_squared > critical_values[dof-1]):
        return True
    else:
        return False

def decisionTree(data, labels, num_classes, heuristic_name, k):
    if(np.all(labels == labels[0])):
        decisions.extend([[k,labels[0]]])
        return
    split_value, best_attr_index = bestSplit(data, labels, num_classes, heuristic_name)
    left_data, left_labels, right_data, right_labels = splitData(data, labels, split_value, best_attr_index)
    if(not shouldSplit(getBucket(left_labels, num_classes), getBucket(right_labels, num_classes))):
        return
    else:
        decisions.extend([[k, split_value, best_attr_index]])
        return [decisionTree(left_data, left_labels, num_classes, heuristic_name, 2*k+1), decisionTree(right_data, right_labels, num_classes, heuristic_name, 2*k+2)]

def getDecisionIndex(decisions, k):
    for i in range(len(decisions)):
        if(decisions[i][0] == k):
            return i
        
def makePrediction(data, decisions, k):
    values = decisions[getDecisionIndex(decisions, k)]
    if(len(values) == 2):
        return values[1]
    else:
        if(data[values[2]] < values[1]):
            return makePrediction(data, decisions, 2*k+1)
        else:
            return makePrediction(data, decisions, 2*k+2)
    
train_set = np.load('dt/train_set.npy')
train_labels = np.load('dt/train_labels.npy')
decisions = []
result = decisionTree(train_set, train_labels, 3, 'info_gain', 0)
       
test_set = np.load('dt/test_set.npy')
test_labels = np.load('dt/test_labels.npy')
print(test_set)
print(test_labels)
test_predictions = np.zeros(test_set.shape[0], dtype=int)
for i in range(test_set.shape[0]):
    test_predictions[i] = makePrediction(test_set[i], decisions, 0)
print(test_predictions)
