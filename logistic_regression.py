import copy, math
import numpy as np
import matplotlib.pyplot as plt
from data1.plt_overfit import overfit_example, output
from data1.lab_utils_common2 import dlc, plot_data, plt_tumor_data, sigmoid, compute_cost_logistic
from data1.plt_quad_logistic import plt_quad_logistic, plt_prob
plt.style.use('./data/deeplearning.mplstyle')
np.set_printoptions(precision = 8)

def compute_cost(X, y, w, b, lambda_ = 0):
    m = X.shape[0]
    f_wb = sigmoid(np.dot(X, w) + b)
    cost = (1 / m) * np.sum((-y) * np.log(f_wb) - (1 - y) * np.log(1 - f_wb)) + (lambda_ / (2 * m)) * np.sum(w ** 2)
    return cost

def compute_gradient(X, y, w, b, lambda_ = 0):
    m = X.shape[0]
    f_wb = sigmoid(np.dot(X, w) + b)
    dj_dw = (1 / m) * np.dot(f_wb - y, X) + (lambda_ / m) * w
    dj_db = (1 / m) * np.sum(f_wb - y)
    return dj_dw, dj_db

def gradient_descent(X, y, w_in, b_in, alpha, num_iters):
    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in
    
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(X, y, w, b)   

        w = w - alpha * dj_dw               
        b = b - alpha * dj_db               

        if i < 100000:
            J_history.append(compute_cost(X, y, w, b))

        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]}   ")

    return w, b, J_history

np.random.seed(1)
X_tmp = np.random.rand(5,3)
y_tmp = np.array([0,1,0,1,0])
w_tmp = np.random.rand(X_tmp.shape[1])
b_tmp = 0.5
lambda_tmp = 0.7
dj_dw_tmp, dj_db_tmp =  compute_gradient(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)

print(f"dj_db: {dj_db_tmp}", )
print(f"Regularized dj_dw:\n {dj_dw_tmp.tolist()}", )