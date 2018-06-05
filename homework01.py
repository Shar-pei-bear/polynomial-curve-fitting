import numpy as np
import matplotlib.pyplot as plt

#coiefficient calculation
def regress(x, x_s, t_s, M, N,lamda = 0):
    order_list =  np.arange((M + 1))
    order_list =  order_list[ :, np.newaxis]
    exponent = np.tile(order_list,[1,N])
    h = np.power(x_s,exponent)
    a =  np.matmul(h, np.transpose(h)) + lamda*np.eye(M+1)
    b =  np.matmul(h, t_s)
    w = np.linalg.solve(a, b) #calculate the coefficent
    
    exponent2 = np.tile(order_list,[1,200])
    h2 = np.power(x,exponent2)
    p = np.matmul(w, h2)
    return p

##task 1
x =  np.linspace(0, 1, 200) 
t =  np.sin(np.pi*x*2)

N = 10
sigma = 0.2;
x_10 = np.linspace(0, 1, N) 
t_10 = np.sin(np.pi*x_10*2) + np.random.normal(0,sigma,N)

plt.figure(1)
plt.plot(x, t, 'g',  x_10, t_10, 'bo',linewidth = 2) 
plt.ylabel('t',rotation='horizontal')
plt.xlabel('x')
plt.savefig('1.png', dpi=300)

##task 2
M = 3
p3_10 =  regress(x, x_10, t_10, M, N)

M = 9
p9_10 =  regress(x, x_10, t_10, M, N)

plt.figure(2)
plt.plot(x, t, 'g',  x_10, t_10, 'bo', x, p3_10, 'r', linewidth = 2) 
plt.ylabel('t',rotation='horizontal')
plt.xlabel('x')
plt.text(0.7, 0.7,'M=3', fontsize=16)
plt.savefig('2.png', dpi=300)

plt.figure(3)
plt.plot(x, t, 'g',  x_10, t_10, 'bo', x, p9_10, 'r', linewidth = 2) 
plt.ylabel('t',rotation='horizontal')
plt.xlabel('x')
plt.text(0.7, 0.7,'M=9', fontsize=16)
plt.savefig('3.png', dpi=300)

##task3
N = 15
x_15 =  np.linspace(0, 1,  N) 
t_15 = np.sin(np.pi*x_15*2) + np.random.normal(0,sigma, N)

M = 9
p9_15 =  regress(x, x_15, t_15, M, N)

N = 100
x_100 =  np.linspace(0, 1,  N) 
t_100 = np.sin(np.pi*x_100*2) + np.random.normal(0,sigma, N)

M = 9
p9_100 =  regress(x, x_100, t_100, M, N)

plt.figure(4)
plt.plot(x, t, 'g',  x_15, t_15, 'bo', x, p9_15, 'r', linewidth = 2) 
plt.ylabel('t',rotation='horizontal')
plt.xlabel('x')
plt.text(0.7, 0.7,'N=15', fontsize=16)
plt.savefig('4.png', dpi=300)

plt.figure(5)
plt.plot(x, t, 'g',  x_100, t_100, 'bo', x, p9_100, 'r', linewidth = 2) 
plt.ylabel('t',rotation='horizontal')
plt.xlabel('x')
plt.text(0.7, 0.7,'N=100', fontsize=16)
plt.savefig('5.png', dpi=300)

##task4
N = 10
M = 9
p9_10 =  regress(x, x_10, t_10, M, N, np.exp(-18))

plt.figure(6)
plt.plot(x, t, 'g',  x_10, t_10, 'bo', x, p9_10, 'r', linewidth = 2) 
plt.ylabel('t',rotation='horizontal')
plt.xlabel('x')
plt.text(0.7, 0.7,'ln$\lambda$ = -18', fontsize=16)
plt.savefig('6.png', dpi=300)
plt.show()
