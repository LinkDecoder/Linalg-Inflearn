import numpy as np

# 가로로
b = np.genfromtxt('pr2_inp1.txt', delimiter=' ', dtype=np.float64)

# 세로로
#b = np.genfromtxt('pr2_inp2.txt', delimiter=' ', dtype=np.float64)

# 가로나 세로나 둘다 1D array로 저장됨을 알 수 있음
print(b.shape)

# 세로로 저장됨
np.savetxt('pr2_out.txt', b, fmt="%0.2f")

