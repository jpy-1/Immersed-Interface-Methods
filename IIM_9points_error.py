import numpy as np
from scipy.optimize import minimize
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.stats import linregress


def phi(x, y):
    return x ** 2 + 4*y ** 2 - 1  # phi(x, y) = 0 对应界面上的点

#beta的定义：
beta_plus=100
beta_minus=1

def fun_beta(x,y):
    if phi(x,y)<=0:
        return beta_minus
    else:
        return beta_plus

#右侧源项
def fun_source_minus(x,y):
    return 0*beta_minus
def fun_source_plus(x,y):
    return (-2*np.sin(x)*np.cos(y))*beta_plus

def fun_source(x,y):
    if phi(x,y)<=0:
        return beta_minus*0
    else:
        return beta_plus*(-2*np.sin(x)*np.cos(y))

#方程的解析解
def fun_analysis_minus(x,y):
    return x**2-y**2

def fun_analysis_plus(x,y): #beta_minus=1
    return np.sin(x)*np.cos(y)

def fun_analysis(x,y):
    if phi(x,y)<=0:
        return fun_analysis_minus(x,y)
    else:
        return fun_analysis_plus(x,y)

#[u]对应的跳跃条件函数
def jump_fun1(x,y):
    return fun_analysis_plus(x,y)-fun_analysis_minus(x,y)


#[beta*u_n]对应的跳跃条件函数
def jump_fun2(x,y):
    h = 1e-6
    grad_phi = np.array([(phi(x + h, y) - phi(x - h, y)) / (2 * h), (phi(x, y + h) - phi(x, y - h)) / (2 * h)])
    unit_grad_phi = grad_phi / np.sqrt(grad_phi[0] ** 2 + grad_phi[1] ** 2)
    grad_u_plus=np.array([(fun_analysis_plus(x+h,y)-fun_analysis_plus(x-h,y))/(2*h),(fun_analysis_plus(x,y+h)-fun_analysis_plus(x,y-h))/(2*h)])
    grad_u_minus=np.array([(fun_analysis_minus(x+h,y)-fun_analysis_minus(x-h,y))/(2*h),(fun_analysis_minus(x,y+h)-fun_analysis_minus(x,y-h))/(2*h)])
    u_n_plus=np.dot(grad_u_plus,unit_grad_phi)
    u_n_minus=np.dot(grad_u_minus,unit_grad_phi)
    return beta_plus*u_n_plus-beta_minus*u_n_minus

# 找到界面上离 (x, y) 最近的点，作为投影点建立局部坐标系
def find_project_point(x,y):
    tol=1e-8
    # 情况1：若当前点已在界面上，直接返回
    if abs(phi(x, y)) < tol:
        return x, y
    # 给定初值：
    r = np.sqrt(x ** 2 + 4 * y ** 2)
    if r == 0:  # 避免除以0
        return 1.0, 0.0  # 界面上的任意点，例如(1, 0)
    x_initial = x / r
    y_initial = y / r

    # 优化目标函数
    def objective_function(point):
        x_star, y_star = point
        return (x_star - x) ** 2 + (y_star - y) ** 2
    # 约束条件
    def constraint_function(point):
        x_star, y_star = point
        return phi(x_star,y_star)

    cons = ({'type': 'eq', 'fun':constraint_function})
    initial_point=np.array([x_initial,y_initial])
    result = minimize(objective_function, initial_point, method='SLSQP', constraints=cons,tol=1e-6)
    return result.x[0],result.x[1]


# 定义判断点 (x, y) 位于界面上哪一侧的函数
def which_side_of_interface(x,y):
    if phi(x,y)<=0:
        return -1 #-侧
    elif phi(x,y)>0:
        return 1 #+侧

# 定义判断点 (x, y) 是否为规则点
def is_regular(x,y,dx,dy):
    tol_interface=max(1e-10, 0.1 * min(dx, dy))
    #1.点在界面上
    if abs(phi(x,y)) < tol_interface:  # 界面上的点视为不规则点
        return False

    # 2. 检查8邻域点是否在界面上
    # 周围四个点的位置 (东、西、南、北)
    points = [
        (x + dx, y),  # 东
        (x - dx, y),  # 西
        (x, y - dy),  # 南
        (x, y + dy),  # 北
        (x + dx, y + dy),  # 东北
        (x - dx, y + dy),  # 西北
        (x - dx, y - dy),  # 西南
        (x + dx, y - dy)
    ]
    phi_values = [phi(px, py) for px, py in points]
    if np.any(np.abs(phi_values) < tol_interface):
        return False

    # 4. 动态符号检查（考虑数值误差）
    ref_sign = np.sign(phi(x, y))
    for p in phi_values:
        if np.sign(p) * ref_sign < -0.5:  # 允许轻微数值波动
            return False
    return True

#获取函数值，给定函数\xi=\chi(\eta):,给定eta,求解\xi，使用牛顿迭代法
def chi(eta,x_star,y_star):
    h = 1e-6
    grad_phi = np.array([(phi(x_star + h, y_star) - phi(x_star - h, y_star)) / (2 * h), (phi(x_star, y_star + h) - phi(x_star, y_star - h)) / (2 * h)])
    unit_grad_phi = grad_phi / np.sqrt(grad_phi[0] ** 2 + grad_phi[1] ** 2)
    transmatrix=np.array([[0,-1],[1,0]])
    unit_cut_phi=np.dot(transmatrix,unit_grad_phi)#切向量
    p0=np.array([x_star,y_star])+eta*unit_cut_phi#初值，因为point=p*+xi*unit_grad_phi*phi+eta*unit_cut_phi
    #目标求解xi,由于fai(x,y)=0,所以迭代求解即满足fai(p0+xi*unit_grad_phi)=0
    xi=0
    tol=1e-12
    max_iter=100
    for i in range(max_iter):
        p=p0+xi*unit_grad_phi
        f=phi(p[0],p[1])
        if abs(f)<tol:
            break
        df=np.dot(np.array([(phi(p[0] + h, p[1]) - phi(p[0] - h, p[1])) / (2 * h), (phi(p[0], p[1] + h) - phi(p[0], p[1] - h)) / (2 * h)]),unit_grad_phi)
        xi=xi-f/df
    return xi

#给定eta后，w'，v'与w''的计算
#给定eta后，界面上该点的w,v函数值
def w_eta(eta,x_star,y_star):
    h = 1e-6
    grad_phi = np.array([(phi(x_star + h, y_star) - phi(x_star - h, y_star)) / (2 * h),
                         (phi(x_star, y_star + h) - phi(x_star, y_star - h)) / (2 * h)])
    unit_grad_phi = grad_phi / np.sqrt(grad_phi[0] ** 2 + grad_phi[1] ** 2)
    transmatrix = np.array([[0, -1], [1, 0]])
    unit_cut_phi = np.dot(transmatrix, unit_grad_phi)  # 切向量
    xi=chi(eta,x_star,y_star)
    point=np.array([x_star,y_star])+eta*unit_cut_phi+xi*unit_grad_phi
    return jump_fun1(point[0],point[1])

def v_eta(eta,x_star,y_star):
    h = 1e-6
    h = 1e-6
    grad_phi = np.array([(phi(x_star + h, y_star) - phi(x_star - h, y_star)) / (2 * h),
                         (phi(x_star, y_star + h) - phi(x_star, y_star - h)) / (2 * h)])
    unit_grad_phi = grad_phi / np.sqrt(grad_phi[0] ** 2 + grad_phi[1] ** 2)
    transmatrix = np.array([[0, -1], [1, 0]])
    unit_cut_phi = np.dot(transmatrix, unit_grad_phi)  # 切向量
    xi = chi(eta, x_star, y_star)
    point = np.array([x_star, y_star]) + eta * unit_cut_phi + xi * unit_grad_phi
    return jump_fun2(point[0], point[1])


#计算局部坐标系夹角
def local_coordinate_system(x_star, y_star):
    h = 1e-6  # 小步长用于数值导数
    phi_x = (phi(x_star + h, y_star) - phi(x_star - h, y_star)) / (2 * h)
    phi_y = (phi(x_star, y_star + h) - phi(x_star, y_star - h)) / (2 * h)
    # 计算法向量
    grad_phi = np.array([phi_x, phi_y])
    norm_grad = np.sqrt(phi_x ** 2 + phi_y ** 2)
    n = grad_phi / norm_grad  # 单位法向量
    # 与 x 轴夹角
    cos_theta = n[0]  # n[0] 是 x 分量
    sin_theta = n[1]  # n[1] 是 y 分量
    return cos_theta, sin_theta

#计算函数对eta的一阶偏导,其中的x,y应为x_star,y_star,即局部坐标系的原点
def partial_eta(x_star,y_star,h=1e-6):
    point=[x_star,y_star]
    v_prime = (v_eta(h, point[0], point[1]) - v_eta(-h, point[0], point[1])) / (2 * h)
    w_prime = (w_eta(h, point[0], point[1]) - w_eta(-h, point[0], point[1])) / (2 * h)
    w_double_prime = (w_eta(h, point[0], point[1]) - 2 * w_eta(0, point[0], point[1]) + w_eta(-h, point[0],
                                                                                              point[1])) / (h ** 2)
    chi_double_prime = (chi(h, point[0], point[1]) - 2 * chi(0, point[0], point[1]) + chi(-h, point[0], point[1])) / (
                h ** 2)
    return w_prime,v_prime,w_double_prime,chi_double_prime

#对gamma应用最大原理
def apply_maximum_principle(A,b,g,dx):
    # 给定初值：
    gamma_h_init=g*dx**2
    # 优化目标函数
    def objective_function(gamma):
        return np.linalg.norm(gamma-g*dx**2)
    #约束条件：
    def constraint_function(gamma):
        return np.dot(A,gamma)-b*dx**2
    bounds=[(0,np.inf)]+[(-np.inf,-1e-10)]+[(0,np.inf)]*7#-1e-10是为了防止gamma_00=0

    constraint={'type': 'eq', 'fun': constraint_function}
    result = minimize(objective_function, gamma_h_init, method='SLSQP',bounds=bounds,constraints=constraint,tol=1e-6)
    if result.success:
        return result.x/dx**2
    else:
        print("Optimization failed:", result.message)
        print("此时系数矩阵为",A/dx**2,"右端项为",b)
        return g

#计算不规则点对应的系数与修正项
def irregular_points_coefficient_minus(x,y,dx,dy):
       #确定模版九点格式
    stencil_cartesian = [
        (x - dx, y),  # u{i-1,j}
        (x, y),      # u_{i,j}
        (x + dx, y),  # u{i+1,j}
        (x, y - dy),  # u{i,j-1}
        (x, y + dy),   # u{i,j+1}
        (x + dx, y - dy),  # 东南
        (x + dx, y + dy),  # 东北
        (x - dx, y - dy),  # 西南
        (x - dx, y + dy)  # 西北
    ]
    stencil_index=[
        (-1,0),
        (0,0),
        (1,0),
        (0,-1),
        (0,1),
        (1, -1),
        (1, 1),
        (-1, -1),
        (-1, 1)
    ]

    stencil_cartesian = np.array(stencil_cartesian)
    x_star, y_star = find_project_point(x, y)
    cos_theta, sin_theta = local_coordinate_system(x_star, y_star)
    w = jump_fun1(x_star, y_star)
    v = jump_fun2(x_star, y_star)
    w_prime, v_prime, w_double_prime, kappa = partial_eta(x_star, y_star)
    f_jump = fun_source_plus(x_star, y_star) - fun_source_minus(x_star, y_star)
    stencil_local=[]
    for x_k,y_k in stencil_cartesian:
        x_local = (x_k - x_star) * cos_theta + (y_k - y_star) * sin_theta  # xi
        y_local = -(x_k - x_star) * sin_theta + (y_k - y_star) * cos_theta  # eta
        stencil_local.append((x_local, y_local))
    stencil_local = np.array(stencil_local)#局部坐标系下的坐标
    #确定正负侧点
    k_plus = []   # 正侧点
    k_minus = []  # 负侧点
    for k, (px, py) in enumerate(stencil_cartesian):
        if which_side_of_interface(px, py) ==-1: # 负侧点或界面上的点
            k_minus.append(k)
        else:
            k_plus.append(k)
    #确定待定系数的系数矩阵与右端项向量
    a_matrix =np.zeros((6,9))
    b_vector=np.zeros(6)
    #定义系数矩阵与右端项向量
    a_matrix[0,k_minus]=1
    a_matrix[0,k_plus]=1
    #b[0]=0

    for k in k_minus:
        a_matrix[1,k]=stencil_local[k][0]
    for k in k_plus:
        a_matrix[1,k]=beta_minus*stencil_local[k][0]/beta_plus
    for k in k_plus:
        a_matrix[1,k]=a_matrix[1,k]-0.5*stencil_local[k][0]**2*((beta_plus-beta_minus)/beta_plus)*kappa
    for k in k_plus:
        a_matrix[1,k]=a_matrix[1,k]+0.5*stencil_local[k][1]**2*((beta_plus-beta_minus)/beta_plus)*kappa
    #b[1]=0

    for k in k_minus:
        a_matrix[2,k]=stencil_local[k][1]
    for k in k_plus:
        a_matrix[2,k]=stencil_local[k][1]
    for k in k_plus:
        a_matrix[2,k]=a_matrix[2,k]+stencil_local[k][0]*stencil_local[k][1]*((beta_plus-beta_minus)/beta_plus)*kappa
    #b[2]=0

    for k in k_minus:
        a_matrix[3,k]=0.5*stencil_local[k][0]**2
    for k in k_plus:
        a_matrix[3,k]=0.5*stencil_local[k][0]**2*beta_minus/beta_plus
    b_vector[3]=beta_minus

    for k in k_minus:
        a_matrix[4,k]=0.5*stencil_local[k][1]**2
    for k in k_plus:
        a_matrix[4,k]=0.5*stencil_local[k][1]**2
    for k in k_plus:
        a_matrix[4,k]=a_matrix[4,k]-0.5*stencil_local[k][0]**2*((beta_plus-beta_minus)/beta_plus)
    b_vector[4]=beta_minus

    for k in k_minus:
        a_matrix[5,k]=stencil_local[k][0]*stencil_local[k][1]
    for k in k_plus:
        a_matrix[5,k]=stencil_local[k][0]*stencil_local[k][1]*beta_minus/beta_plus
    #b[5]=0
    #求解待定系数
    g=np.zeros(9)#逼近系数g
    g[0]=fun_beta(x-dx,y)/dx**2
    g[1]=-2*fun_beta(x,y)*(1/dx**2+1/dy**2)
    g[2]=fun_beta(x+dx,y)/dx**2
    g[3]=fun_beta(x,y-dy)/dy**2
    g[4]=fun_beta(x,y+dy)/dy**2
    coefficient=apply_maximum_principle(a_matrix,b_vector,g,dx)

    #计算修正项
    #a的计算：
    a_2=a_4=a_6=a_8=a_10=a_12=0
    for k in k_plus:
        a_2=a_2+coefficient[k]
        a_4=a_4+coefficient[k]*stencil_local[k][0]
        a_6=a_6+coefficient[k]*stencil_local[k][1]
        a_8=a_8+coefficient[k]*0.5*stencil_local[k][0]**2
        a_10=a_10+coefficient[k]*0.5*stencil_local[k][1]**2
        a_12=a_12+coefficient[k]*stencil_local[k][0]*stencil_local[k][1]
    C=a_2*w+a_12*v_prime/beta_plus+(a_6+a_12*kappa)*w_prime \
       +a_10*w_double_prime+1/beta_plus*(a_4+(a_8-a_10)*kappa)*v \
       +a_8*(f_jump/beta_plus-w_double_prime)
    return coefficient,C,stencil_index


def irregular_points_coefficient_plus(x, y, dx, dy):
    stencil_cartesian = [
        (x - dx, y),  # u{i-1,j}
        (x, y),  # u_{i,j}
        (x + dx, y),  # u{i+1,j}
        (x, y - dy),  # u{i,j-1}
        (x, y + dy),  # u{i,j+1}
        (x + dx, y - dy),  # 东南
        (x + dx, y + dy),  # 东北
        (x - dx, y - dy),  # 西南
        (x - dx, y + dy)  # 西北
    ]
    stencil_index = [
        (-1, 0),
        (0, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 1),
        (-1, -1),
        (-1, 1)
    ]
    stencil_cartesian = np.array(stencil_cartesian)
    x_star, y_star = find_project_point(x, y)
    cos_theta, sin_theta = local_coordinate_system(x_star, y_star)
    w = jump_fun1(x_star, y_star)
    v = jump_fun2(x_star, y_star)
    w_prime, v_prime, w_double_prime,kappa = partial_eta(x_star, y_star)
    f_jump = fun_source_plus(x_star, y_star) - fun_source_minus(x_star, y_star)
    stencil_local = []
    for x_k, y_k in stencil_cartesian:
        x_local = (x_k - x_star) * cos_theta + (y_k - y_star) * sin_theta  # xi
        y_local = -(x_k - x_star) * sin_theta + (y_k - y_star) * cos_theta  # eta
        stencil_local.append((x_local, y_local))
    stencil_local = np.array(stencil_local)  # 局部坐标系下的坐标
    # 确定正负侧点
    k_plus = []  # 正侧点
    k_minus = []  # 负侧点
    for k, (px, py) in enumerate(stencil_cartesian):
        if which_side_of_interface(px, py) ==-1:  # 负侧点或界面上的点
            k_minus.append(k)
        else:
            k_plus.append(k)
    # 确定待定系数的系数矩阵与右端项向量
    a_matrix = np.zeros((6, 9))
    b_vector = np.zeros(6)
    # 确定系数矩阵与右端项向量
    a_matrix[0, k_minus] = 1
    a_matrix[0, k_plus] = 1
    # b[0]=0

    for k in k_minus:
        a_matrix[1, k] = stencil_local[k][0]*beta_plus/beta_minus
    for k in k_plus:
        a_matrix[1, k] = stencil_local[k][0]
    for k in k_minus:
        a_matrix[1, k] = a_matrix[1, k] + 0.5 * stencil_local[k][0] ** 2 * ((beta_plus-beta_minus)/beta_minus) * kappa
    for k in k_minus:
        a_matrix[1, k] = a_matrix[1, k] - 0.5 * stencil_local[k][1] ** 2 * ((beta_plus-beta_minus)/beta_minus) * kappa
    # b[1]=0

    for k in k_minus:
        a_matrix[2, k] = stencil_local[k][1]
    for k in k_plus:
        a_matrix[2, k] = stencil_local[k][1]
    for k in k_minus:
        a_matrix[2, k] = a_matrix[2, k] - stencil_local[k][0] * stencil_local[k][1] * ((beta_plus-beta_minus)/beta_minus) * kappa
    # b[2]=0

    for k in k_minus:
        a_matrix[3, k] = 0.5 * stencil_local[k][0] ** 2*beta_plus/beta_minus
    for k in k_plus:
        a_matrix[3, k] = 0.5 * stencil_local[k][0] ** 2
    b_vector[3] = beta_plus

    for k in k_minus:
        a_matrix[4, k] = 0.5 * stencil_local[k][1] ** 2
    for k in k_plus:
        a_matrix[4, k] = 0.5 * stencil_local[k][1] ** 2
    for k in k_minus:
        a_matrix[4, k] = a_matrix[4, k] + 0.5 * stencil_local[k][0] ** 2 * ((beta_plus-beta_minus)/beta_minus)
    b_vector[4] = beta_plus

    for k in k_minus:
        a_matrix[5, k] = stencil_local[k][0] * stencil_local[k][1]*beta_plus/beta_minus
    for k in k_plus:
        a_matrix[5, k] = stencil_local[k][0] * stencil_local[k][1]
    # b[5]=0
    # 求解待定系数
    g = np.zeros(9)  # 逼近系数g
    g[0] = fun_beta(x - dx, y) / dx ** 2
    g[1] = -2 * fun_beta(x, y) * (1 / dx ** 2 + 1 / dy ** 2)
    g[2] = fun_beta(x + dx, y) / dx ** 2
    g[3] = fun_beta(x, y - dy) / dy ** 2
    g[4] = fun_beta(x, y + dy) / dy ** 2
    coefficient = apply_maximum_principle(a_matrix, b_vector, g,dx)

    # 计算修正项
    # a的计算：
    a_1 = a_3 = a_5 = a_7 = a_9 = a_11 = 0
    for k in k_minus:
        a_1 = a_1 + coefficient[k]
        a_3 = a_3 + coefficient[k] * stencil_local[k][0]
        a_5 = a_5 + coefficient[k] * stencil_local[k][1]
        a_7 = a_7 + coefficient[k] * 0.5 * stencil_local[k][0] ** 2
        a_9 = a_9 + coefficient[k] * 0.5 * stencil_local[k][1] ** 2
        a_11 = a_11 + coefficient[k] * stencil_local[k][0] * stencil_local[k][1]

    C = -a_1 * w - a_3 * v / beta_minus - a_5 * w_prime + \
        a_7 * (-kappa * v / beta_minus + w_double_prime - f_jump / beta_minus) \
        + a_9 * (kappa * v / beta_minus - w_double_prime) \
        + a_11 * (-kappa * w_prime - v_prime / beta_minus)

    return coefficient, C, stencil_index

def IIM_9_point(N):
    x = np.linspace(-2, 2, N)
    y = np.linspace(-2, 2, N)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # 系数矩阵与右侧向量的构造：
    a_matrix = lil_matrix(((N - 2) * (N - 2), (N - 2) * (N - 2)))
    b_vector = np.zeros((N - 2) * (N - 2))
    for i in range(N - 2):
        for j in range(N - 2):
            x_point = x[i + 1]
            y_point = y[j + 1]
            # 判断是否为规则点
            if is_regular(x_point, y_point, dx, dy):
                beta_point = fun_beta(x_point, y_point)
                a_matrix[i * (N - 2) + j, i * (N - 2) + j] = -2 * (1 / dx ** 2 + 1 / dy ** 2) * beta_point
                if i > 0:
                    a_matrix[i * (N - 2) + j, (i - 1) * (N - 2) + j] = beta_point / dx ** 2
                if i < N - 3:
                    a_matrix[i * (N - 2) + j, (i + 1) * (N - 2) + j] = beta_point / dx ** 2
                if j > 0:
                    a_matrix[i * (N - 2) + j, i * (N - 2) + j - 1] = beta_point / dy ** 2
                if j < N - 3:
                    a_matrix[i * (N - 2) + j, i * (N - 2) + j + 1] = beta_point / dy ** 2
                if i == 0 and j == 0:
                    b_vector[i * (N - 2) + j] = fun_source(x_point, y_point) - \
                                                beta_point * fun_analysis(x[0],
                                                                          y[1]) / dx ** 2 - beta_point * fun_analysis(
                        x[1], y[0]) / dy ** 2
                elif i == 0 and j == N - 3:
                    b_vector[i * (N - 2) + j] = fun_source(x_point, y_point) - \
                                                beta_point * fun_analysis(x[0], y[
                        N - 2]) / dx ** 2 - beta_point * fun_analysis(x[1], y[N - 1]) / dy ** 2
                elif i == N - 3 and j == 0:
                    b_vector[i * (N - 2) + j] = fun_source(x_point, y_point) - \
                                                beta_point * fun_analysis(x[N - 1],
                                                                          y[1]) / dx ** 2 - beta_point * fun_analysis(
                        x[N - 2], y[0]) / dy ** 2
                elif i == N - 3 and j == N - 3:
                    b_vector[i * (N - 2) + j] = fun_source(x_point, y_point) - \
                                                beta_point * fun_analysis(x[N - 1], y[
                        N - 2]) / dx ** 2 - beta_point * fun_analysis(x[N - 2], y[N - 1]) / dy ** 2

                elif i == 0:
                    b_vector[i * (N - 2) + j] = fun_source(x_point, y_point) - beta_point * fun_analysis(x[0],
                                                                                                         y_point) / dx ** 2
                elif i == N - 3:
                    b_vector[i * (N - 2) + j] = fun_source(x_point, y_point) - beta_point * fun_analysis(x[N - 1],
                                                                                                         y_point) / dx ** 2
                elif j == 0:
                    b_vector[i * (N - 2) + j] = fun_source(x_point, y_point) - beta_point * fun_analysis(x_point,
                                                                                                         y[0]) / dy ** 2
                elif j == N - 3:
                    b_vector[i * (N - 2) + j] = fun_source(x_point, y_point) - beta_point * fun_analysis(x_point, y[
                        N - 1]) / dy ** 2
                else:
                    b_vector[i * (N - 2) + j] = fun_source(x_point, y_point)
            else:
                if which_side_of_interface(x_point, y_point) == -1:  # 在-侧或界面上
                    coefficient_minus, C_minus, stencil_index_minus = irregular_points_coefficient_minus(x_point,
                                                                                                         y_point, dx,
                                                                                                         dy)
                    for k_stencil, (i_stencil, j_stencil) in enumerate(stencil_index_minus):
                        a_matrix[i * (N - 2) + j, (i + i_stencil) * (N - 2) + j + j_stencil] = coefficient_minus[
                            k_stencil]
                    b_vector[i * (N - 2) + j] = fun_source(x_point, y_point) + C_minus
                else:  # 在+侧
                    coefficient_plus, C_plus, stencil_index_plus = irregular_points_coefficient_plus(x_point, y_point,
                                                                                                     dx, dy)
                    for k_stencil, (i_stencil, j_stencil) in enumerate(stencil_index_plus):
                        a_matrix[i * (N - 2) + j, (i + i_stencil) * (N - 2) + j + j_stencil] = coefficient_plus[
                            k_stencil]
                    b_vector[i * (N - 2) + j] = fun_source(x_point, y_point) + C_plus

    # 转换为 CSC 格式以提高求解效率
    a_matrix = a_matrix.tocsc()
    #
    # 求解线性方程组
    u_approx = spsolve(a_matrix, b_vector)

    # m1 = pyamg.ruge_stuben_solver(a_matrix)
    # # solve linear system using AMG
    # u_approx = m1.solve(b_vector, tol=1e-10)

    # 解析解
    u_analysis = np.zeros((N - 2) * (N - 2))
    for i in range(N - 2):
        for j in range(N - 2):
            x_point = x[i + 1]
            y_point = y[j + 1]
            u_analysis[i * (N - 2) + j] = fun_analysis(x_point, y_point)

    # 无穷范数误差计算
    error = np.linalg.norm(u_analysis - u_approx, ord=np.inf)
    return error

# List of N values to test
# N_values =np.logspace(4,9,12,base=2)
N_values = [20,29,44,65,144,215,320]
h_values = [1 / N for N in N_values]
errors = []

# Compute relative errors for different N values
for N in N_values:
    N=int(N)
    error = IIM_9_point(N)
    errors.append(error)
    print(f'N = {N}, h = {1 / N:.8f},  Error = {error:.10f}')

# Calculate convergence order
log_h = np.log(h_values)
log_E = np.log(errors)
slope, intercept, r_value, p_value, std_err = linregress(log_h, log_E)
print(f'Convergence order (slope): {slope:.4f}')
print(r_value)