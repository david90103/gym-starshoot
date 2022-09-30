import numpy as np


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

arr = []

with open('grep.txt', 'r') as f:
    for line in f:
        arr.append(float(line.split()[3]))

arr = moving_average(np.array(arr), 1)

c = 1
with open('grep.txt', 'w') as f:
    for n in arr:
        f.write(str(c) + ' ' + str(n) + '\n')
        c += 1

