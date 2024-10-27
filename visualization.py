import matplotlib.pyplot as plt
import numpy as np
import torch

from cases.case_triangle import Triangle
from pinnsfomer import PinnsFormer

# 弹性体属性
l = 2.0
e = 20.1
nu = 0.3
q0 = 15


def generate_fig(x, y, sigma_x, sigma_y, tau_xy, title):
    # 计算冯米塞斯等效应力
    sigma_v = []
    for i in range(len(x)):
        sigma_t = np.mat([[sigma_x[i], tau_xy[i]], [tau_xy[i], sigma_y[i]]])
        lam, _ = np.linalg.eig(sigma_t)
        sigma_v_ = np.sqrt((lam[0] ** 2) + (lam[1] ** 2) - (lam[0] * lam[1]))
        sigma_v.append(sigma_v_)
    sigma_v = np.array(sigma_v)

    # 可视化结果
    fig, ax = plt.subplots(2, 2, figsize=(12, 12))

    sc = ax[0][0].scatter(x, y, c=sigma_x, cmap='viridis')
    fig.colorbar(sc, ax=ax[0][0], label='σ_x')
    ax[0][0].set_title('σ_x Stress Distribution')

    sc = ax[0][1].scatter(x, y, c=sigma_y, cmap='viridis')
    fig.colorbar(sc, ax=ax[0][1], label='σ_y')
    ax[0][1].set_title('σ_y Stress Distribution')

    sc = ax[1][0].scatter(x, y, c=tau_xy, cmap='viridis')
    fig.colorbar(sc, ax=ax[1][0], label='τ_xy')
    ax[1][0].set_title('τ_xy Stress Distribution')

    sc = ax[1][1].scatter(x, y, c=sigma_v, cmap='viridis')
    fig.colorbar(sc, ax=ax[1][1], label='σ_v')
    ax[1][1].set_title('σ_v Stress Distribution')

    fig.suptitle(title)
    return fig


def main():
    # 网格划分
    nx = 100
    ny = 100

    # 生成三角形内部点和边界条件
    elastic_body = Triangle(e, nu, l, q0)
    x, y = elastic_body.geometry(nx, ny)
    bc = elastic_body.boundary_conditions(x, y)
    u_gt, v_gt, sigma_x_gt, sigma_y_gt, tau_xy_gt = elastic_body.ground_truth(x, y)

    x = torch.tensor(x, dtype=torch.float32).view(-1, 1).unsqueeze(0).requires_grad_(True)
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1).unsqueeze(0).requires_grad_(True)
    bc = torch.tensor(bc, dtype=torch.float32).unsqueeze(0)

    # pinn = PinnsFormer(d_model=64, d_hidden=64, n=4, heads=2, e=e, nu=nu)
    pinn = torch.load("./pinn.pth").cpu()
    u, v, epsilon_x, epsilon_y, gamma_xy, sigma_x, sigma_y, tau_xy = pinn(x, y, bc)

    x = x.view(-1).detach().numpy()
    y = y.view(-1).detach().numpy()
    sigma_x = sigma_x.view(-1).detach().numpy()
    sigma_y = sigma_y.view(-1).detach().numpy()
    tau_xy = tau_xy.view(-1).detach().numpy()

    err_sigma_x = np.abs(sigma_x - sigma_x_gt)
    err_sigma_y = np.abs(sigma_y - sigma_y_gt)
    err_tau_xy = np.abs(tau_xy - tau_xy_gt)

    generate_fig(x, y, sigma_x, sigma_y, tau_xy, "Predicted Stress Distribution")
    plt.show()
    generate_fig(x, y, sigma_x_gt, sigma_y_gt, tau_xy_gt, "Ground Truth Stress Distribution")
    plt.show()
    generate_fig(x, y, err_sigma_x, err_sigma_y, err_tau_xy, "Error Stress Distribution")
    plt.show()


if __name__ == '__main__':
    main()
