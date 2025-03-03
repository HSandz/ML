import math
import numpy as np
import matplotlib.pyplot as plt
from data1.lab_utils_uni import plt_house_x, plt_contour_wgrad, plt_divergence, plt_gradients
plt.style.use('./data/deeplearning.mplstyle')

def compute_model_output(x, w, b):
    
    # Normal loop
    # m = x.shape[0]
    # f_wb = np.zeros(m)
    # for i in range(m):
    #     f_wb[i] = w * x[i] + b
    # return f_wb
    
    # Vectorized:
    return w * x + b
    

def compute_cost(x, y, w, b):
    m = x.shape[0]
    
    # Normal loop
    # cost = 0
    # for i in range(m):
    #     f_wb = w * x[i] + b
    #     cost = cost + (f_wb - y[i])**2
    # total_cost = 1 / (2 * m) * cost
    #
    # return total_cost
    
    # Vectorized:
    f_wb = compute_model_output(x, w, b)
    cost = (1 / (2 * m)) * np.sum((f_wb - y) ** 2)
    return cost

def compute_gradient(x, y, w, b):
    m = x.shape[0]
    
    # Normal loop
    # dj_dw = 0
    # dj_db = 0
    #
    # for i in range(m):
    #     f_wb = w * x[i] + b
    #     dj_dw_i = (f_wb - y[i]) * x[i]
    #     dj_db_i = f_wb - y[i]
    #     dj_db += dj_db_i
    #     dj_dw += dj_dw_i
    # dj_dw /= m
    # dj_db /= m
    #
    # return dj_dw, dj_db
    
    # Vectorized:
    f_wb = compute_model_output(x, w, b)
    dj_dw = (1 / m) * np.dot(f_wb - y, x)
    dj_db = (1 / m) * np.sum(f_wb - y)
    
    return dj_dw, dj_db

def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
    J_history = []
    p_history = []
    b = b_in
    w = w_in
    
    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x, y, w, b)
        
        b -= alpha * dj_db
        w -= alpha * dj_dw
        
        if i < 100000:
            J_history.append(cost_function(x, y, w, b))
            p_history.append([w,b])
        
        if i % math.ceil(num_iters/10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")
    
    return w, b, J_history, p_history

x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

# Plot gradients
# plt_gradients(x_train,y_train, compute_cost, compute_gradient)
# plt.show()

w_init = 0
b_init = 0

iterations = 10000
tmp_alpha = 1.0e-2

w_final, b_final, J_hist, p_hist = gradient_descent(x_train, y_train, w_init, b_init, tmp_alpha,
                                                    iterations, compute_cost, compute_gradient)
print(f"(w,b) found by gradient descent: ({w_final:8.4f}, {b_final:8.4f})")
