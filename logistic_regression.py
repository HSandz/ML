import copy, math
import numpy as np
import matplotlib.pyplot as plt
from data.lab_utils_common2 import dlc, plot_data, plt_tumor_data, sigmoid, compute_cost_logistic
from data.plt_quad_logistic import plt_quad_logistic, plt_prob
plt.style.use('./data/deeplearning.mplstyle')

def compute_gradient_logistic(X, y, w, b):
    m = X.shape[0]
    f_wb = sigmoid(np.dot(X, w) + b)
    dj_dw = (1 / m) * np.dot(f_wb - y, X)
    dj_db = (1 / m) * np.sum(f_wb - y)
    return dj_dw, dj_db

def gradient_descent(X, y, w_in, b_in, alpha, num_iters):
    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in
    
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient_logistic(X, y, w, b)   

        w = w - alpha * dj_dw               
        b = b - alpha * dj_db               

        if i < 100000:
            J_history.append(compute_cost_logistic(X, y, w, b))

        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]}   ")

    return w, b, J_history

X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])

# # Plot data
# fig,ax = plt.subplots(1,1,figsize=(4,4))
# plot_data(X_train, y_train, ax)
#
# ax.axis([0, 4, 0, 3.5])
# ax.set_ylabel('$x_1$', fontsize=12)
# ax.set_xlabel('$x_0$', fontsize=12)
# plt.show()

w_tmp  = np.zeros_like(X_train[0])
b_tmp  = 0.
alph = 0.1
iters = 10000

w_out, b_out, _ = gradient_descent(X_train, y_train, w_tmp, b_tmp, alph, iters) 
print(f"\nupdated parameters: w:{w_out}, b:{b_out}")

# Plot the result of gradient descent
fig,ax = plt.subplots(1,1,figsize=(5,4))

# plot the probability 
plt_prob(ax, w_out, b_out)
# Original data
ax.set_ylabel(r'$x_1$')
ax.set_xlabel(r'$x_0$')   
ax.axis([0, 4, 0, 3.5])
plot_data(X_train,y_train,ax)
# Decision boundary
x0 = -b_out/w_out[0]
x1 = -b_out/w_out[1]
ax.plot([0,x0],[x1,0], c=dlc["dlblue"], lw=1)
plt.show()