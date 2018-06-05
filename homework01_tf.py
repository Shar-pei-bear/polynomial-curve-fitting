import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.01)
  return tf.Variable(initial)

def H_mat(N,M,x_s, x):
  order_list =  np.arange((M + 1))
  order_list =  order_list[ :, np.newaxis]
  
  exponent = np.tile(order_list,[1,N])
  exponent2 = np.tile(order_list,[1,200])
  
  h1 = np.power(x_s,exponent  )
  h2 = np.power(x,   exponent2)

  return h1, h2

# Model parameters
M = 9
N = 10
lamda = np.exp(-18);

W = weight_variable([1, M + 1])

# Model input and output
x = tf.placeholder(tf.float32)
linear_model = tf.matmul(W,  x) 
y = tf.placeholder(tf.float32)

# loss
loss = tf.reduce_sum(tf.square(linear_model - y) )+ lamda* tf.reduce_sum(tf.square(W)) # sum of the squares

# optimizer
#optimizer = tf.train.GradientDescentOptimizer(0.04)
optimizer =tf.train.AdamOptimizer(0.1)
train = optimizer.minimize(loss)

# training data
x_200 =  np.linspace(0, 1, 200) 
t_200 =  np.sin(np.pi*x_200*2)

sigma = 0.2;
x_10 = np.linspace(0, 1, N)
t_10 = np.sin(np.pi*x_10*2) + np.random.normal(0,sigma,N)
h_10_3, h_200_3 = H_mat(N,M,x_10, x_200)

# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
loss_new = 0.0
for i in range(100000):
  if i%100 == 0:
      loss_old = loss_new
      loss_new = sess.run(loss, {x:h_10_3, y:t_10})
      if (abs(loss_old - loss_new) < 1e-8):
        break
      print("index: %s loss: %s"%(i, loss_new))
  sess.run(train, {x:h_10_3, y:t_10})

# evaluate training accuracy
curr_W, curr_loss  = sess.run([W, loss], {x:h_10_3, y:t_10})
p3_10 = np.squeeze(np.matmul(curr_W, h_200_3))
plt.figure(1)
plt.plot(x_200, t_200, 'g',  x_10, t_10, 'bo', x_200, p3_10, 'r', linewidth = 2) 
plt.ylabel('t',rotation='horizontal')
plt.xlabel('x')
plt.text(0.7, 0.7,'ln$\lambda$ = -18', fontsize=16)
plt.show()
