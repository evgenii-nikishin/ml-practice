import numpy as np


def encoding1(x):
    pos = np.where(np.diff(x) != 0)[0]
    counts = np.diff(np.concatenate(([-1], pos, [x.shape[0] - 1]), axis=1))
    return np.append(x[pos], x[x.shape[0] - 1]), counts


def encoding2(x):
    values = []
    counts = []
    if (x.shape[0] != 0):
        count = 1
        for i in range(1, len(x)):
            if x[i] == x[i-1]:
                count += 1
            else:
                values.append(x[i-1])
                counts.append(count)
                count = 1
        else:
            values.append(x[len(x) - 1])
            counts.append(count)
    return np.array(values), np.array(counts)


def encoding3(x):
    values = x.copy()
    counts = []
    idx = 0
    cur_count = 1
    while idx < values.shape[0] - 1:
        if values[idx] == values[idx+1]:
            cur_count += 1
            values = np.delete(values, idx+1, axis=0)
        else:
            idx += 1
            counts.append(cur_count)
            cur_count = 1
    counts.append(cur_count)
    return values, np.array(counts)