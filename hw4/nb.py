import numpy as np


def vocabulary(data):
    """
    Creates the vocabulary from the data.
    :param data: List of lists, every list inside it contains words in that sentence.
                 len(data) is the number of examples in the data.
    :return: Set of words in the data
    """
    vocab = set([])
    for item in data:
        vocab.update(item)
    return vocab

def estimate_pi(train_labels):
    """
    Estimates the probability of every class label that occurs in train_labels.
    :param train_labels: List of class names. len(train_labels) is the number of examples in the training data.
    :return: pi. pi is a dictionary. Its keys are class names and values are their probabilities.
    """
    return {x:train_labels.count(x)/len(train_labels) for x in train_labels}

def theta_helper(train_data, labels, word, label, vocab_size):
        a,b = 0,0
        for i in range(len(train_data)):
            if(labels[i] == label):
                a += train_data[i].count(word)
            b += len(train_data[i])
        return (1+a)/(vocab_size+b)
    
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
    classes = set(train_labels)
    theta = {class_name:{} for class_name in classes}
    
    for class_label in classes:
        word_dict = {word:theta_helper(train_data, train_labels, word, class_label, len(vocab)) for word in vocab}
        total_value = 0
        for word in vocab:
            total_value += word_dict[word]
        k = 1/total_value
        for word in vocab:
            word_dict[word] *= k
        theta[class_label] = word_dict
        
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
    
