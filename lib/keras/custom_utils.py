import random
import numpy as np
def random_float(low, high,N):
    arr=[]
    for i in range(N):
        arr.append(random.random()*(high-low) + low)
    return np.array(arr)