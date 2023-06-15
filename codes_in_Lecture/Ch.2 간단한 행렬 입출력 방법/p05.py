import numpy as np

# real 행렬
A1 = np.array( [ [1,2.5,3], [-1,-2,-1.5], [4,5.5,6] ], dtype=np.float64 )

# complex 행렬
A2 = np.array( [ [1-2j,3+1j,1], [1+2j,2-1j,7] ], dtype=np.complex128 )

# floating 포맷으로 소수점 2자리까지 output1.txt에 A1을 출력하기, 구분기호는 콤마(,)
np.savetxt('output1.txt', A1, fmt='%0.2f', delimiter=',')

# scientific 포맷 예시, A2를 output2.txt에 출력
np.savetxt('output2.txt', A2, fmt='%0.4e', delimiter=',')
