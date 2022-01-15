import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

train_set = np.load('svm/task3/train_set.npy')
train_set = train_set.reshape(train_set.shape[0], train_set.shape[1]*train_set.shape[2])
train_labels = np.load('svm/task3/train_labels.npy')

train_size = int(train_set.shape[0]*0.8)
random_idx = np.arange(train_set.shape[0])
np.random.shuffle(random_idx)
train_set, validation_set = train_set[random_idx[:train_size]], train_set[random_idx[train_size:]]
train_labels, validation_labels = train_labels[random_idx[:train_size]], train_labels[random_idx[train_size:]]

test_set = np.load('svm/task3/test_set.npy')
test_set = test_set.reshape(test_set.shape[0], test_set.shape[1]*test_set.shape[2])
test_labels = np.load('svm/task3/test_labels.npy')

kernels = ['linear', 'rbf', 'poly', 'sigmoid']
c_values = [0.01, 0.1, 1, 10, 100]
accuracies = []

clfs = []
print('-'*44)
for kernel in kernels:
    for c_value in c_values:
        clf = SVC(kernel=kernel, C=c_value)
        clf = clf.fit(train_set, train_labels)
        clfs.append(clf)

        predicted_labels = clf.predict(validation_set)

        accuracy = 100*np.bincount(np.abs(validation_labels-predicted_labels))[0]/np.size(validation_labels)
        accuracies.append(accuracy)
        
        print('|kernel: ' + kernel + ' '*(7-len(kernel)) + '| C: ' + str(c_value) + ' '*(4-len(str(c_value))) + '| accuracy = %' + str(accuracy) +'|')
        print('-'*44)
        
max_idx = accuracies.index(max(accuracies))
max_kernel = kernels[max_idx//5]
max_c = c_values[max_idx%5]

print('kernel: ' + max_kernel + ', C: ' + str(max_c) +', validation accuracy = %' + str(max(accuracies)))

clf = clfs[max_idx]
predicted_labels = clf.predict(test_set)
accuracy = 100*np.bincount(np.abs(test_labels-predicted_labels))[0]/np.size(test_labels)

print('accuracy on test set: ' + str(accuracy))