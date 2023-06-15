import numpy as np
from print_lecture import print_custom as prt

# real 행렬
A1 = np.array( [ [1,2.5,3], [-1,-2,-1.5], [4,5.5,6] ], dtype=np.float64 )

# complex 행렬
A2 = np.array( [ [1-2j,3+1j,1], [1+2j,2-1j,7] ], dtype=np.complex128 )

# floating 포맷으로 출력 예시
prt(A1, fmt='%0.4f', delimiter='.')

# 보기편하게 공백줄
print()

# scientific 포맷으로 complex 행렬 출력 예시
prt(A2,fmt='%0.2e', delimiter=',')