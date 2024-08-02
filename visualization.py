import matplotlib.pyplot as plt
import numpy as np
import torch

from data_generation import generate_points, generate_bc
from pinnsfomer import PinnsFormer


def generate_fig(x, y, sigma_x, sigma_y, tau_xy):
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
    return fig


def main():
    l = 2.0
    q0 = 15

    x, y = generate_points(l)
    bc = generate_bc(x, y, l, q0)

    x = torch.tensor(x, dtype=torch.float32).view(-1, 1).unsqueeze(0)
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1).unsqueeze(0)
    bc = torch.tensor(bc, dtype=torch.float32).unsqueeze(0)

    pinn = PinnsFormer(d_model=64, d_hidden=64, N=4, heads=2)
    # pinn = torch.load("./pinn.pth")
    sigma_x, sigma_y, tau_xy = pinn(x, y, bc)

    x = x.view(-1).detach().numpy()
    y = y.view(-1).detach().numpy()
    sigma_x = sigma_x.view(-1).detach().numpy()
    sigma_y = sigma_y.view(-1).detach().numpy()
    tau_xy = tau_xy.view(-1).detach().numpy()

    generate_fig(x, y, sigma_x, sigma_y, tau_xy)
    plt.show()


if __name__ == '__main__':
    main()
