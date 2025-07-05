#[u]!=0 and [\beta u_x]!=0
import numpy as np
import matplotlib.pyplot as plt
import pyamg
import time
import scipy.sparse as sp
from scipy.stats import linregress
# 设置 Matplotlib 字体以支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows 系统常用字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
def beta(t):
    if 0<=t<=1/2:
        result=1
    else:
        result=2
    return result
beta_plus=2 #beta>1/2
beta_minus=1 #beta<1/2

# define f(x)
def fun(t):# f(x)
    result=12*t**2
    return result

def fun_analytical(t):# analytical solution
    if 0<=t<=1/2:
        result=t**4-1.375*t
    else:
        result=t**4/2+0.3125*t+0.1875
    return result
# def fun_analytical(t):# analytical solution
#     if 0<t<=1/2:
#         result=t**4/beta_minus
#     else:
#         result=t**4/beta_plus+(1/beta_minus-1/beta_plus)*(1/2)**4
#     return result

# generate grid
N=100
x=np.linspace(0,1,N+1)
h=x[1]-x[0]

#define boundary conditions
u_0=0
u_end=1
# u_0=0
# u_end=1/beta_plus+(1/beta_minus-1/beta_plus)*(1/2)**4

index_irregular1=np.searchsorted(x, 0.5, side='right')-1 #x_j index j

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

#irregular points
w=1 #[u]=1
v=2 #[\beat u_x]=2
D_1=h**2+(beta_plus-beta_minus)*(x[index_irregular1-1]-1/2)*(x[index_irregular1]-1/2)/(2*beta_minus)
D_2=h**2-(beta_plus-beta_minus)*(x[index_irregular1+2]-1/2)*(x[index_irregular1+1]-1/2)/(2*beta_plus)

gamma_11=(beta_minus-(beta_plus-beta_minus)*(x[index_irregular1]-1/2)/h)/D_1
gamma_12=(-2*beta_minus+(beta_plus-beta_minus)*(x[index_irregular1-1]-1/2)/h)/D_1
gamma_13=beta_plus/D_1

gamma_21=beta_minus/D_2
gamma_22=(-2*beta_plus+(beta_plus-beta_minus)*(x[index_irregular1+2]-1/2)/h)/D_2
gamma_23=(beta_plus-(beta_plus-beta_minus)*(x[index_irregular1+1]-1/2)/h)/D_2

A[index_irregular1-1,index_irregular1-2]=gamma_11
A[index_irregular1-1,index_irregular1-1]=gamma_12
A[index_irregular1-1,index_irregular1]=gamma_13
A[index_irregular1,index_irregular1-1]=gamma_21
A[index_irregular1,index_irregular1]=gamma_22
A[index_irregular1,index_irregular1+1]=gamma_23

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
C1=gamma_13*(w+(x[index_irregular1+1]-1/2)*v/beta_plus)
C2=gamma_21*(-w-(x[index_irregular1]-1/2)*v/beta_minus)
B[index_irregular1-1]=fun(x[index_irregular1])+C1
B[index_irregular1]=fun(x[index_irregular1+1])+C2
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
