import numpy as np
import time
if __name__ == '__main__':
    x = np.random.randn(100, 200)
    max_x = x.max()
    start_time = time.time()
    # numerator = np.exp(x - max_x)
    out = np.exp(x - max_x) / np.einsum('ik->i', np.exp(x - max_x))[:, np.newaxis]
    print(time.time() - start_time)