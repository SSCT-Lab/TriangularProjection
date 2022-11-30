import numpy as np

def get_data_by_label(X, Y, label):
    idx_arr = np.where(Y == label)
    y = Y[idx_arr]
    x = X[idx_arr]
    return (x, y)
