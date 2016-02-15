# -*- coding: utf-8 -*-

import numpy as np

def calc_entropy(X):
    """Returns entropy of labels' distribution"""
    
    #Проверка корректности входных значений
    if not isinstance(X, np.ndarray):
        raise TypeError('Input vector must be given by a numpy array')
    
    if X.ndim != 1:
        raise TypeError('Input vector must be one-dimensional')
    
    if X.size == 0:
        raise TypeError('Input vector must be non-empty')
    
    #Вычисления
    _, counts = np.unique(X, return_counts=True)
    probs = counts / X.size
    
    return -np.sum(probs * np.log2(probs))


#Альтернативная реализация из интернета
def calc_entropy2(X):
    """ Computes entropy of 0-1 vector. """
    n_labels = len(X)

    if n_labels <= 1:
        return 0

    counts = np.bincount(X)
    probs = counts[np.nonzero(counts)] / n_labels
    n_classes = len(probs)

    if n_classes <= 1:
        return 0
    return - np.sum(probs * np.log2(probs))