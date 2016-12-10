from index import split_into_periods
import numpy as np

result = np.array([0, 0, 0], dtype='int')
split_into_periods([1.0, 2.0, 3.0, 8.0, 9.0, 11.0], 3.0, result)

print(result)
