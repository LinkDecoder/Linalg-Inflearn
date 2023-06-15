import numpy as np

# complex 행렬도 차이가 없이 똑같음, 입력 파일의 형태만 주의해야하고 dtype만 주의 할것
# 아래는 entry 구분기호가 띄워쓰기 인 입력 파일, python 실행하는 곳에 있어야함.
# dtype을 np.complex128로 바꾸면 complex128 행렬로 저장됨
A = np.genfromtxt('p03_inp1.txt', delimiter=' ', dtype=np.complex128)

# 아래는 entry 구분기호가 콤마(,)인 입력 파일
#A = np.genfromtxt('p03_inp2.txt', delimiter=',', dtype=np.complex128)

print(A)