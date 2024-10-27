import numpy as np
import torch
from torch.optim import LBFGS

from cases.case_triangle import Triangle
from loss_func import PinnLoss
from pinnsfomer import PinnsFormer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 弹性体属性
l = 2.0
e = 201
nu = 0.3

# 外力Q的取值范围
q_min = 10
q_max = 20


def random_choice(dataset_num, batch_size):
    indices = np.zeros(batch_size)
    batch_gird = dataset_num / batch_size
    for i in range(batch_size):
        begin = batch_gird * i
        end = batch_gird * (i + 1)
        indices[i] = np.random.randint(begin, end)
    return indices


def main():
    # 网格划分
    nx = 40
    ny = 40

    # 训练参数
    dataset_num = 1000
    epochs = 300
    batch_size = 10

    # 生成三角形内部点
    elastic_body = Triangle(e, nu, l, q_min)
    x, y = elastic_body.discretize(nx, ny)

    # 生成边界条件数据
    bc = []
    for q0 in np.arange(q_min, q_max, (q_max - q_min) / dataset_num):
        elastic_body.set_load(q0)
        bc_ = elastic_body.boundary_conditions(x, y)
        bc.append(bc_)
    bc = torch.tensor(bc, dtype=torch.float32, requires_grad=False)

    # 转换为PyTorch张量
    x = torch.tensor(x, dtype=torch.float32).view(-1, 1)
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    x_copy = x.repeat(dataset_num, 1, 1).requires_grad_(False)
    y_copy = y.repeat(dataset_num, 1, 1).requires_grad_(False)

    # 初始化网络和优化器
    pinn = PinnsFormer(d_model=64, d_hidden=64, N=2, heads=2, E=e, nu=nu).to(device)
    optimizer = LBFGS(pinn.parameters(), lr=1e-1, line_search_fn='strong_wolfe')
    loss_func = PinnLoss(e)

    # 训练
    for epoch in range(epochs):
        # 随机选择一个batch
        indices = random_choice(dataset_num, batch_size)
        x_batch = x_copy[indices].to(device).requires_grad_(True)
        y_batch = y_copy[indices].to(device).requires_grad_(True)
        bc_batch = bc[indices].to(device)

        def closure():
            # 前向传播
            u, v, epsilon_x, epsilon_y, gamma_xy, sigma_x, sigma_y, tau_xy = pinn(x_batch, y_batch, bc_batch)
            # 计算损失
            loss = loss_func(elastic_body, x_batch, y_batch, u, v, sigma_x, sigma_y, tau_xy, bc_batch)
            print("Epoch: ", epoch, ", Loss: ", loss.cpu().detach().numpy())
            # 梯度下降
            optimizer.zero_grad()
            loss.backward()
            return loss

        optimizer.step(closure)

    # 保存模型
    torch.save(pinn, "./pinn.pth")


if __name__ == "__main__":
    main()
