import numpy as np


def generate_points(l, point_num=50):
    x_train, y_train = np.meshgrid(np.linspace(0, l, point_num), np.linspace(0, l, point_num))
    x_train = x_train.flatten()
    y_train = y_train.flatten()
    train_mask = y_train <= x_train
    x_train = x_train[train_mask]
    y_train = y_train[train_mask]
    return x_train, y_train


def generate_bc(x, y, l, q0):
    # 生成边界条件
    bc = np.zeros([x.shape[0], 5])
    for i in range(x.shape[0]):
        x_idx = x[i]
        y_idx = y[i]

        # Face AB
        if y_idx == 0:
            qx = -x_idx / l * q0
            bc[i] = np.array([0, -1, 0, qx, 1])

        # Face BC
        elif x_idx == l:
            tx = q0 * (2 * y_idx / l - 1)
            ty = y_idx / l * q0
            bc[i] = np.array([1, 0, tx, ty, 1])

        # Face AC
        elif y_idx == x_idx:
            l = -1 / np.sqrt(2)
            m = 1 / np.sqrt(2)
            bc[i] = np.array([l, m, 0, 0, 1])

        else:
            bc[i] = np.array([0, 0, 0, 0, 0])
    return bc
