import numpy as np
data = np.array([[1,2, 5],[2,3, 6],[3,4, 7],[4,5, 8],[5, 6, 9], [6,7, 10], [7, 8, 11], [8, 9, 12]])

print('3 to -1')
print(data[3:, :-1])

print('3 to 1 ')
print(data[3:, 1:])