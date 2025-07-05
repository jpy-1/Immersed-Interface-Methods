import numpy as np
import matplotlib.pyplot as plt
import pyamg
import time
import scipy.sparse as sp
from scipy.stats import linregress
# 设置 Matplotlib 字体以支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows 系统常用字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


alpha=2/3
R=0.05
beta_plus=5
beta_minus=1
def beta(t):
    if 0<=t<=alpha:
        result=1
    else:
        result=5
    return result

# define f(x)
def fun(t):# f(x)
    result=6*t
    return result

def fun_analytical(t):# analytical solution
    if 0<=t<=alpha:
        result=t**3/beta_minus
    else:
        result=t**3/beta_plus+(1/beta_minus-1/beta_plus)*alpha**3+3*R*alpha**2
    return result

# generate grid
N=32
x=np.linspace(0,1,N+1)
h=x[1]-x[0]

#define boundary conditions
u_0=fun_analytical(0)
u_end=fun_analytical(1)

index_irregular1=np.searchsorted(x, alpha, side='right')-1 #x_j index j

#constrcut matrix A
A = sp.lil_matrix((N-1, N-1))
A[0,0]=-2*beta_minus/h**2
A[0,1]=beta_minus/h**2
A[N-2,N-2]=-2*beta_plus/h**2
A[N-2,N-3]=beta_plus/h**2
for i in range(1,index_irregular1-1):
    A[i,i]=-2*beta_minus/h**2
    A[i,i+1]=1*beta_minus/h**2
    A[i,i-1]=1*beta_minus/h**2
for i in range(index_irregular1+1,N-2):
    A[i,i]=-2*beta_plus/h**2
    A[i,i+1]=1*beta_plus/h**2
    A[i,i-1]=1*beta_plus/h**2

#求解gamma:
gamma_1A=np.zeros((3,3))
gamma_1b=np.zeros(3)
gamma_2A=np.zeros((3,3))
gamma_2b=np.zeros(3)
gamma_1A[0,0]=1
gamma_1A[0,1]=1
gamma_1A[0,2]=1
gamma_1A[1,0]=x[index_irregular1-1]-alpha
gamma_1A[1,1]=x[index_irregular1]-alpha
gamma_1A[1,2]=R*beta_minus+(x[index_irregular1+1]-alpha)*beta_minus/beta_plus
gamma_1A[2,0]=1/2*(x[index_irregular1-1]-alpha)**2
gamma_1A[2,1]=1/2*(x[index_irregular1]-alpha)**2
gamma_1A[2,2]=1/2*(x[index_irregular1+1]-alpha)**2*beta_minus/beta_plus
gamma_1b[2]=beta_minus

#########################################################
gamma_2A[0,0]=1
gamma_2A[0,1]=1
gamma_2A[0,2]=1
gamma_2A[1,0]=-R*beta_plus+(x[index_irregular1]-alpha)*beta_plus/beta_minus
gamma_2A[1,1]=(x[index_irregular1+1]-alpha)
gamma_2A[1,2]=(x[index_irregular1+2]-alpha)
gamma_2A[2,0]=1/2*(x[index_irregular1]-alpha)**2*beta_plus/beta_minus
gamma_2A[2,1]=1/2*(x[index_irregular1+1]-alpha)**2
gamma_2A[2,2]=1/2*(x[index_irregular1+2]-alpha)**2
gamma_2b[2]=beta_plus
#########################################################
gamma_1=np.linalg.solve(gamma_1A,gamma_1b)
gamma_2=np.linalg.solve(gamma_2A,gamma_2b)

A[index_irregular1-1,index_irregular1-2]=gamma_1[0]
A[index_irregular1-1,index_irregular1-1]=gamma_1[1]
A[index_irregular1-1,index_irregular1]=gamma_1[2]
A[index_irregular1,index_irregular1-1]=gamma_2[0]
A[index_irregular1,index_irregular1]=gamma_2[1]
A[index_irregular1,index_irregular1+1]=gamma_2[2]

# 转换为 CSR 格式
A_csr = A.tocsr()

#construct B
#regular points
B=np.zeros(N-1)
B[0]=fun(x[1])-u_0*beta_minus/h**2
B[N-2]=fun(x[N-1])-u_end*beta_plus/h**2
for i in range(1,index_irregular1-1):
    B[i]=fun(x[i+1])
for i in range(index_irregular1+1,N-2):
    B[i]=fun(x[i+1])

#irregular points
B[index_irregular1-1]=fun(x[index_irregular1])
B[index_irregular1]=fun(x[index_irregular1+1])

# 使用 PyAMG 求解
ml = pyamg.ruge_stuben_solver(A_csr)  # 使用 CSR 格式的矩阵
u_approx = ml.solve(B, tol=1e-10)  # 求解线性方程组

#analytical solution
u_analytical=np.zeros(N-1)
for i in range(N-1):
    u_analytical[i]=fun_analytical(x[i+1])

#error
# error=np.linalg.norm(u_approx-u_analytical)/np.linalg.norm(u_analytical)
error = np.linalg.norm(u_approx - u_analytical, np.inf)
print("error=",error)

#plot
plt.plot(x[1:N],u_approx,'r*',label='数值解')
plt.plot(x[1:N],u_analytical,'b-',label='解析解')
plt.legend()
plt.xlabel('x')
plt.ylabel('u')
plt.grid(True)
plt.show()