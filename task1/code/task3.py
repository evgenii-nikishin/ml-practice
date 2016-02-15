import numpy as np
from collections import defaultdict


def check_multisets1(x, y):
    return np.array_equal(np.sort(x), np.sort(y))


def check_multisets2(x, y):
    flag = True
    d1 = defaultdict(int)
    d2 = defaultdict(int)
    for key in x:
        d1[key] += 1
    for key in y:
        d2[key] += 1
    if len(d1) != len(d2):
        return False
    else:
        for i in d1.keys():
            if d1[i] != d2[i]:
                flag = False
    return flag

def check_multisets3(x, y):
    if x.shape[0] != y.shape[0]:
        return False
    l1 = sorted(list(x))
    l2 = sorted(list(y))
    for i in range(len(l1)):
        if l1[i] != l2[i]:
            return False
    return True