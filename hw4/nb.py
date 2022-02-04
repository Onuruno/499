import numpy as np

def vocabulary(data):
    """
    Creates the vocabulary from the data.
    :param data: List of lists, every list inside it contains words in that sentence.
                 len(data) is the number of examples in the data.
    :return: Set of words in the data
    """
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    vocab = set([])
    for item in data:
        for word in item:
            for elem in word:
                if(elem in punc):
                    word = word.replace(elem, "")
            vocab.add(word)
    return vocab

def estimate_pi(train_labels):
    """
    Estimates the probability of every class label that occurs in train_labels.
    :param train_labels: List of class names. len(train_labels) is the number of examples in the training data.
    :return: pi. pi is a dictionary. Its keys are class names and values are their probabilities.
    """
    return {x:train_labels.count(x)/len(train_labels) for x in train_labels}
    
def theta_helper(train_data, vocab):
    word_frequency = {word:1 for word in vocab}
    total_word = 0
    for data in train_data:
        for word in data:
            punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
            for elem in word:
                if(elem in punc):
                    word = word.replace(elem, "")
            if(word in vocab):
                word_frequency[word] +=1
                total_word +=1
    for word in word_frequency:
        word_frequency[word] /= (len(vocab)+total_word)
    
    return word_frequency
        
def estimate_theta(train_data, train_labels, vocab):
    """
    Estimates the probability of a specific word given class label using additive smoothing with smoothing constant 1.
    :param train_data: List of lists, every list inside it contains words in that sentence.
                       len(train_data) is the number of examples in the training data.
    :param train_labels: List of class names. len(train_labels) is the number of examples in the training data.
    :param vocab: Set of words in the training set.
    :return: theta. theta is a dictionary of dictionaries. At the first level, the keys are the class names. At the
             second level, the keys are all the words in vocab and the values are their estimated probabilities given
             the first level class name.
    """
    classes = list(set(train_labels))
    theta = {class_name:{} for class_name in classes}
    
    for class_label in classes:
        indices = [i for i, x in enumerate(train_labels) if x == class_label]
        class_data = [train_data[i] for i in indices]
        theta[class_label] = theta_helper(class_data, vocab)   
    return theta

def test(theta, pi, vocab, test_data):
    """
    Calculates the scores of a test data given a class for each class. Skips the words that are not occurring in the
    vocabulary.
    :param theta: A dictionary of dictionaries. At the first level, the keys are the class names. At the second level,
                  the keys are all of the words in vocab and the values are their estimated probabilities.
    :param pi: A dictionary. Its keys are class names and values are their probabilities.
    :param vocab: Set of words in the training set.
    :param test_data: List of lists, every list inside it contains words in that sentence.
                      len(test_data) is the number of examples in the test data.
    :return: scores, list of lists. len(scores) is the number of examples in the test set. Every inner list contains
             tuples where the first element is the score and the second element is the class name.
    """
    scores = []
    
    for data in test_data:
        data_arr = []
        for class_name in pi:
            class_value = 0
            for word in data:
                if(word in vocab and word in theta[class_name]):
                    class_value += np.log(theta[class_name][word])
            class_value += np.log(pi[class_name])
            data_arr.append((class_value, class_name))
        scores.append(data_arr)
    
    return scores

with open('nb_data/train_set.txt', encoding='UTF-8') as file:       #train_set path
    lines = file.read().splitlines()
    
train_data = [row[1:].split() for row in lines]

with open('nb_data/test_set.txt', encoding='UTF-8') as file:        #test_set path
    lines = file.read().splitlines()
    
test_data = [row[1:].split() for row in lines]

with open('nb_data/train_labels.txt') as file:          #train_label path
    train_labels = file.read().splitlines()
    
with open('nb_data/test_labels.txt') as file:           #test_label path
    test_labels = file.read().splitlines()

vocab = vocabulary(train_data)
print(len(vocab))

pi = estimate_pi(train_labels)
theta = estimate_theta(train_data, train_labels, vocab)
scores = test(theta, pi, vocab, test_data)

counter = 0
for i in range(len(scores)):
    value = scores[i][0][0]
    class_name = scores[i][0][1]
    for item in scores[i]:
        if(item[0]>value):
            value = item[0]
            class_name = item[1]
    if(train_labels[i] == class_name):
        counter +=1

print('accuracy = %' + str(100*counter/len(train_labels)))
