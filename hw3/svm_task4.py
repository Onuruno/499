import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

train_set = np.load('svm/task4/train_set.npy')
train_set = train_set.reshape(train_set.shape[0], train_set.shape[1]*train_set.shape[2])
train_labels = np.load('svm/task4/train_labels.npy')

test_set = np.load('svm/task4/test_set.npy')
test_set = test_set.reshape(test_set.shape[0], test_set.shape[1]*test_set.shape[2])
test_labels = np.load('svm/task4/test_labels.npy')

##########################################################################################

clf = SVC(kernel='rbf', C=1)
clf = clf.fit(train_set, train_labels)

predicted_labels = clf.predict(test_set)
accuracy = 100*np.bincount(np.abs(test_labels-predicted_labels))[0]/np.size(test_labels)

print('accuracy of test set with original data: %' + str(accuracy))

##########################################################################################

train_set2 = np.copy(train_set)
train_labels2 = np.copy(train_labels)

minority_class_value = np.argmin(np.bincount(train_labels2))
ratio = np.size(train_labels2)//np.bincount(train_labels2)[minority_class_value]-1

minority_idx = train_labels2[train_labels2 == minority_class_value]
minority_set = train_set2[minority_idx]
minority_labels = train_labels2[minority_idx]
for i in range(ratio-1):
    train_set2 = np.append(train_set2, minority_set, axis=0)
    train_labels2 = np.append(train_labels2,minority_labels)

clf = clf.fit(train_set2, train_labels2)    
predicted_labels = clf.predict(test_set)
accuracy = 100*np.bincount(np.abs(test_labels-predicted_labels))[0]/np.size(test_labels)

print('accuracy of test set with oversampled minority data: %' + str(accuracy))

##########################################################################################

train_set3 = np.copy(train_set)
train_labels3 = np.copy(train_labels)

minority_class_value = np.argmin(np.bincount(train_labels3))
majority_class_value = np.argmax(np.bincount(train_labels3))
minority_size = np.bincount(train_labels3)[minority_class_value]

majority_idx = np.where(train_labels3 == majority_class_value)[0]
delete_idx = majority_idx[minority_size:]

train_set3 = np.delete(train_set3, delete_idx, axis=0)
train_labels3 = np.delete(train_labels3, delete_idx, axis=0)

clf = clf.fit(train_set3, train_labels3)
predicted_labels = clf.predict(test_set)
accuracy = 100*np.bincount(np.abs(test_labels-predicted_labels))[0]/np.size(test_labels)

print('accuracy of test set with undersampled majority data: %' + str(accuracy))

#########################################################################################

clf = SVC(kernel='rbf', C=1, class_weight='balanced')
clf = clf.fit(train_set, train_labels)
predicted_labels = clf.predict(test_set)
accuracy = 100*np.bincount(np.abs(test_labels-predicted_labels))[0]/np.size(test_labels)

print('accuracy of test set with balanced class weights: %' + str(accuracy))