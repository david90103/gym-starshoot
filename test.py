import random 
import time
import numpy as np

big = [[1 for _ in range(100)] for _ in range(1000000)]
big = np.array(big)

print("init finish")

start = time.time()

idx = random.sample(range(0, len(big)), 64)
print(big[idx])

print(time.time() - start)
