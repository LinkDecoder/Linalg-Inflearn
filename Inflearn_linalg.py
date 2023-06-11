import numpy as np
from scipy import linalg

## 1강: 행렬 및 벡터 표현법
# np.array(Mat, dtype)
# Mat.shape
# A[i][j] = A[i, j]
# Mat = Mat.astype(dtype) 명시적 타입변환
# Mat + mat 행렬 간 연산 시 묵시적 타입변환
print('\n\n 1st Class-----------------------')
A_mat = np.array([[1,2],[3,4]], dtype=np.float64)
print('\n A_mat:\n', A_mat)
print('\n A_mat_shape:\n', A_mat.shape)



## 2강: 간단한 행렬 입출력 방법
# np.genfromtxt('filename', delimiter='', dtype)
# np.savetxt('filename', Mat, fmt='%0.4f', delimiter='')
print('\n\n 2nd Class-----------------------')



## 3강: 행렬 관련 편리한 기능
# np.eye(row, col, band_id, dtype) band에 1채우기
# np.identity(val, dtype) identity 행렬 생성
# np.tri(row, col, band_id, dtype) 1로 채워지는 하삼각행렬 생성
# np.zeros(shape) 0행렬 생성, tuple타입 shape 사용
# np.ones(shape) 1행렬 생성, tuple타입 shape 사용
# np.full(shape, value) val로 채움, tuple타입 shape 사용 
# np.random.rand(row, col) 0~1 사이의 무작위 값으로 채워지는 행렬 생성
print('\n\n 3rd Class-----------------------')
A_eye = np.eye(2,3,k=-1, dtype=np.complex128)
print('\n A_eye:\n', A_eye)
B_tri = np.tri(2,3)
print('\n B_tri:\n', B_tri)
C_rand = np.random.rand(2,2)
print('\nRandom mat:\n', C_rand)



## 4강: 행렬 기본 조작 (1)
# np.trace(Mat) trace계산
# np.triu(Mat) 행렬에서 위삼각행렬 생성
# np.tril(Mat) 행렬에서 하삼각행렬 생성
# np.diag(Mat or 1Darr) 대각선 부분을 1D화 or 대각행렬 생성
# np.diagflat(1Darr) 1D 행렬을 대각선에 두고 2D 정사각행렬 생성
# Mat.flatten = np.raven(a) 1D화 한 뒤 카피(flatten)/같은 메모리(ravel)
print('\n\n 4rd Class-----------------------')
a = np.array([[1,2,3], [4,5,6], [7,8,9]], dtype=np.float64)
b_reshaped = np.reshape(a, (1,9))
print('\n b_reshaped:\n', b_reshaped)
a_complex = a+1j*a
b = np.diag(a)
c = np.diag(np.diag(a), k=1)
d = np.diagflat(np.diag(a))
print('\n Diag of Mat a:\n', c)
print('\n Diagflat of Mat a:\n', d)
val_trace = np.trace(a)
print('\n a\'s trace:\n', val_trace)



## 5강: 행렬 기본 조작 (2)
# np.hstack(matrix tuple) 수평으로 쌓음, row 개수가 동일해야함
# np.vstack(matrix tuple) 수직으로 쌓음, col 개수가 동일해야함
# Mat.transpose or Mat.T 같은 메모리를 참조하는 transpose 행렬 반환
# Mat.real, Mat.imag, Mat.conjugate
# np.matmul(A, B) or A @ B
# np.vdot(u, v) complex인 경우 u_conj dot v 임에 유의
# np.dot(u, v) complex인 경우 그냥 dot을 계산함
print('\n\n 5rd Class-----------------------')
a = np.array([[1,2],[3,4]], dtype=np.float64)
b = np.array([[-1, -2],[-3,-4]], dtype=np.float64)
A_hstacked = np.hstack((a, b))
print('\n hstracked:\n', A_hstacked)
ab_complex = a + 1j*b
ab_real = np.copy(ab_complex.real)
ab_imag = np.copy(ab_complex.imag)
print('\n Mat:\n', ab_complex)
print('\n Mat.real:\n', ab_real)
print('\n Mat.imag:\n', ab_imag)





## 6강: 행렬 기본 조작 (3)
# A+-*/B, A*/b, b*/A
# idx = [1,0,3,2], A[idx, :] 행렬의 row 순서 변경
# A[]

