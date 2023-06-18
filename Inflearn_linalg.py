import numpy as np
from scipy import linalg
import timeit
from print_lecture import print_custom as prt
from custom_band import read_banded
from custom_band import matmul_banded
from custom_band import read_banded_h
from custom_band import matmul_banded_h
from custom_sp import matmul_toeplitz
from custom_sp import matmul_circulant
from custom_decomp import perm_from_piv

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
print('\n\n 6rd Class-----------------------')



## 7강: 일반 행렬
# linalg.det(Mat) Lapack: zgetrf, dgetrf
# linalg.inv(Mat) Lapack: getrf, getri
# linalg.solve(A, b, assum_a="gen") Ax=b 해결, assum_a={gen, sym, her, pos}
# gen: A의 성질을 모르는 경우, LU decomposition 사용, Lapack: gesv
# sym: A가 symmetric matrix인 경우, Diagonal pivoting method 사용, Lapack: sysv
# her: A가 Hermitian matrix인 경우, Diagonal pivoting method 사용, Lapack: hesv
# pos: A가 positive definite인 경우, Cholesky decomposition 사용, posv
# linalg.solve_triangular(A, b, lower=False) True: Lower matrix, False: Upper matrix, Lapack: trtrs
# np.allclose(A@x, b) 두 값이 충분히 비슷하면 True, 아니면 False 반환
# np.allclose(x, y)는 |x-y|<=(eps1 + eps2*|y|), eps1 = 1e-8, eps2 = 1e-5로 결정
print('\n\n 7rd Class-----------------------')
A1 = np.array([[1, 5, 0], [2, 4, -1], [0, -2, 0]])
A2 = np.array([[1, -4, 2], [-2, 8, -9], [-1, 7, 0]])
det1 = linalg.det(A1)
print('\n det:\n', det1)
A1_inv = linalg.inv(A1)
print('\n inv:\n', A1_inv)
b = np.ones((3,1))
b_lowerTri = np.ones((4, 1))
A_singular = np.array([[1, 3, 4], [-4, 2, -6], [-3, -2, -7]])
A_gen = np.array([[0, 1, 2], [1, 0, 3], [4, -3, 8]])
A_symmetric = np.array([[1, 2, 1], [2, 1, 3], [1, 3, 1]])
A_symmetric_complex = np.array([[1, 2-1j, 1+2j], [2-1j, 1, 3], [1+2j, 3, 1]])
A_hermitian = np.array([[2, 2+3j, 10-2j], [2-3j, 1, 3], [10+2j, 3, 2]])
A_positivedefinite = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
A_lowerTri = np.array([[1, 0, 0, 0], [1, 4, 0, 0], [5, 0, 1, 0], [8, 1, -2, 2]])
x_gen = linalg.solve(A_gen, b, assume_a="gen")
x_symmetric = linalg.solve(A_symmetric, b, assume_a="sym")
x_symmetric_complex = linalg.solve(A_symmetric_complex, b, assume_a="sym")
# x_hermitian = linalg.solve(A_hermitian, b, "her")
x_positivedefinite = linalg.solve(A_positivedefinite, b, "pos")
x_lowerTri = linalg.solve_triangular(A_lowerTri, b_lowerTri, lower=True)
print('\n np.allclose:\n', np.allclose(A_gen@x_gen, b), np.allclose(A_symmetric@x_symmetric, b), \
      np.allclose(A_positivedefinite@x_positivedefinite, b), np.allclose(A_lowerTri@x_lowerTri, b_lowerTri))



## 8강: 밴드 행렬
# linalg.solve_banded((lbw, ubw), Mat_Band, b), Ax=b의 해
# LU decomposition Lapack: gbsv
# tridiagonal solver Lapack: gtsv
# linalg.solveh_banded(A_bandh, b, lower=False), Positive definite band 행렬인 경우
# Cholesky decomposition Lapack: pbsv
# LDLT decomposition Lapack: ptsv, Positive definite tridiagonal인 경우
print('\n\n 8rd Class-----------------------')
b = np.ones((5,))
A1_band = read_banded("./Matrix_in_txt/p04_inp1.txt", (2,1), dtype=np.float64, delimiter=" ")
A2_band = read_banded("./Matrix_in_txt/p06_inp2.txt", (1,1), dtype=np.float64, delimiter=" ")
x1_band = linalg.solve_banded((2,1), A1_band, b)
x2_band = linalg.solve_banded((1,1), A2_band, b)
print('\n np.allclose:\n', np.allclose(matmul_banded((2,1), A1_band, x1_band), b))
print('\n np.allclose:\n', np.allclose(matmul_banded((1,1), A2_band, x2_band), b))
A1_band_h = read_banded_h("./Matrix_in_txt/p10_inp1.txt", 1, dtype=np.complex128, delimiter=" ", lower=False)
b = np.ones((4,))
x1_band_h = linalg.solveh_banded(A1_band_h, b, lower=False)
print('\n np.allclose:\n', np.allclose(matmul_banded_h(1, A1_band_h, x1_band_h), b))



## 9강: 특수 행렬
# linalg.solve_toeplitz((c, r), b), c, r: 1D vector, Levinson-Durbin recurson
# linalg.toeplitz(c, r), toeplitz 행렬 생성
# linalg.solve_circulatn(c, b), FFT로 문제 해결, c는 column임에 유의
# linalg.circulant(c), circulant 행렬 생성
print('\n\n 9rd Class-----------------------')
c = np.array([1, 3, 6, 10])
r = np.array([1, -1, -2, -3])
b = np.ones((4,), dtype=np.float64)
x_toeplitz = linalg.solve_toeplitz((c, r), b)
print('\n np.allclose:\n', np.allclose(matmul_toeplitz((c, r), x_toeplitz), b))
c = np.array([2, -1, 0, 1, 0, 0, 1])
b = np.ones((7,))
x_circulant = linalg.solve_circulant(c, b)
print('\n np.allclose:\n', np.allclose(matmul_circulant(c, x_circulant), b))



## 10강: 동시에 여러 식 풀기
# X = linalg.solve(A, B, assume_a="gen"), B에도 행렬을 넣으면 X도 행렬로 반환
# 모든 solve 함수가 위와 같음
print('\n\n 10rd Class-----------------------')



## 11강: 고유치 계산
# 보통 QR Algorithm 사용, 일부 Jacobi algorithm
# A=QR, A1=R1Q1, A1=Q2R2, A2=R2Q2....Ak는 triangular 행렬로 수렴, eigenvalue가 동일
# QR decompositino 방법: 보통 Householder method, 일부 Givens reduction
# Numerical recipes chapter 2&11 참고
print('\n\n 11rd Class-----------------------')



## 12강: 고유치 계산(일반 행렬)
# (eigvals, eigvecs) = linalg.eig(A, M, right=True), Ax=lamda*x or Ax=lamda*M*x, right=false이면 eigenvalue만 반환
# (eigvals, eigvecs) = linalg.eigh(A, M, eigvals_only=False), A가 symmetric/hermitian, M이 positive definite, Lapack: (syevr, heevr, sygvd, hevgd)
# 1)reduction to tridiagonal form, Householder 2)dqds algorithm/Relatively Robust Representations, Lapack: 1)sytrd, hetrd 2)stemr, ormtr, unmtr
# Linear Algebra and its application, B.N.Parlett & I.S.Dhillon
# 실행 시간=time_end-time_start, time_start = timeit.default_timer(), time_end = timeit.default_timer()
print('\n\n 12rd Class-----------------------')
A_eig = np.array([[0, -1], [1, 0]]) # eigvals = (i, -i), eigvecs=[[1, -i], [1, i]]
(eigvals, eigvecs) = linalg.eig(A_eig)
print('\n (eigvals, eigvecs):\n', eigvals, '\n', eigvecs)
v1 = eigvecs[:, 0]
print('\n Norm of eigvecs:\n', linalg.norm(v1))
comp1 = A_eig@eigvecs
comp2 = eigvecs*eigvals
print('\n np.allclose:\n', np.allclose(comp1, comp2))
eigvals_comp = linalg.eig(A_eig, right=False)
print('\n np.allclose:\n', np.allclose(eigvals, eigvals_comp))
A_eigh = np.array([[6, 3, 1, 5], [3, 0, 5, 1], [1, 5, 6, 2], [5, 1, 2, 2]])
(eighvals, eighvecs) = linalg.eigh(A_eigh)
print('\n (eighvals, eighvecs):\n', eighvals, '\n', eighvecs)
comp1_eigh = A_eigh@eighvecs
comp2_eigh = eighvecs*eighvals
print('\n np.allclose:\n', np.allclose(comp1_eigh, comp2_eigh))



## 13강: 고유치 계산(밴드 행렬)
# (eigvals, eigvecs) = linalg.eig_banded(A_baded_half, lower=False), A는 symmetric/hermitian upper banded matrix
print('\n\n 13rd Class-----------------------')
A_banded_symmetric = np.array([[1, 5, 2, 0], [5, 2, 5, 2], [2, 5, 3, 5], [0, 2, 5, 4]])
A_banded_symmetric_upper = np.array([[0, 0, 2, 2], [0, 5, 5, 5], [1, 2, 3, 4]])
(eigvals_band, eigvecs_band) = linalg.eig_banded(A_banded_symmetric_upper)
print('\n (eigvals_band, eigvecs_band):\n', eigvals_band, '\n', eigvecs_band)
comp1 = A_banded_symmetric@eigvecs_band
comp2 = eigvecs_band*eigvals_band
print('\n np.allclose:\n', np.allclose(comp1, comp2))



## 14강: Power method
# 주어진 eigenvalue 중 가장 큰 값 및 이에 해당하는 eigenvector를 구하는 알고리즘
# convergence ratio: lamda2/lamda1, 알고리즘이 수렴하는 속도를 나타내는 수
# Inverse power method: A 대신 inv(A)로 power method를 적용 시 eigenvalue 중 가장 작은 값 및 이에 해당하는 eigenvector 계산
# inv(A)를 직접 계산하여 사용하지 않음, LU decompostion하여 적용
print('\n\n 14rd Class-----------------------')
A_pm = np.array([[6, 5], [1, 2]], dtype=np.float64) #lamda=(7, 1)
x_iter = np.array([1,0], dtype=np.float64)
for k in range(1, 100):
    x_new = A_pm@x_iter
    x_new_normalized = x_new/linalg.norm(x_new)
    mu = np.vdot(x_new, A_pm@x_new)/np.vdot(x_new, x_new)
    x_iter = x_new



## 15강: 행렬 분해(1)
# (P, L, U) = linalg.lu(A), A=PLU, 실제 계산에서는 다른 함수를 사용함에 유의
# (lu, piv) = linalg.lu_factor(A), lu: L과 U를 한 행렬에 저장, piv: 1D array, row interchange 정보 저장, i row는 piv[i]와 interchange
print('\n\n 15rd Class-----------------------')
A_lu = np.array([[2, 4, -1, 5, -2], [-4, -5, 3, -8, 1], [2, -5, -4, 1, 8], [-6, 0, 7, -3, 1]])
(P, L, U) = linalg.lu(A_lu)
print('\n P, L, U:\n', P)
print()
prt(L, fmt="%0.2f")
print()
prt(U, fmt="%0.2f")
print('\n A_lu:\n', P@L@U)
(lu, piv) = linalg.lu_factor(A_lu)
perm_A_lu = perm_from_piv(piv)
print('\n A_lu:')
prt((L@U)[perm_A_lu, :], fmt="%0.1f")









