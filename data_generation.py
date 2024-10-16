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
    bc = np.zeros([x.shape[0], 7])

    # 生成边界条件
    # 边界条件编码格式：[l, m, tx, ty, u, v, mask]
    # l, m: 法向量
    # tx, ty: 对于力边界，tx, ty为力的分量
    # u, v: 对于位移边界，u, v为位移的分量
    for i in range(x.shape[0]):
        x_idx = x[i]
        y_idx = y[i]

        # Face AB，受到向下的分布载荷
        if y_idx == 0:
            qx = -x_idx / l * q0
            bc[i] = np.array([0, -1, 0, qx, 0, 0, 2])

        # Face BC，固定在墙上
        elif x_idx == l:
            bc[i] = np.array([0, 0, 0, 0, 0, 0, 1])

        # Face AC，不受力
        elif y_idx == x_idx:
            l_ = -1 / np.sqrt(2)
            m_ = 1 / np.sqrt(2)
            bc[i] = np.array([l_, m_, 0, 0, 0, 0, 2])

        # 其他内部点无边界条件
        else:
            bc[i] = np.array([0, 0, 0, 0, 0, 0, 0])
    return bc
