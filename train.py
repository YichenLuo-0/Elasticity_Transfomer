import numpy as np
import torch
from torch.optim import LBFGS

from cases.case_triangle import Triangle
from loss_func import PinnLoss
from pinnsfomer import PinnsFormer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 弹性体属性
l = 2.0
e = 20.1
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
    # Mesh size of the elastic body
    nx = 40
    ny = 40

    # Training parameters
    dataset_num = 1000
    epochs = 300
    batch_size = 10

    # Initialize the elastic body
    elastic_body = Triangle(e, nu, l, q_min)
    x, y = elastic_body.geometry(nx, ny)

    # Initialize the boundary conditions and ground truth lists
    bc = []
    u_gt = []
    v_gt = []
    sigma_x_gt = []
    sigma_y_gt = []
    tau_xy_gt = []

    # Generate boundary conditions and ground truth
    for q0 in np.arange(q_min, q_max, (q_max - q_min) / dataset_num):
        elastic_body.set_load(q0)
        bc_ = elastic_body.boundary_conditions(x, y)
        u_gt_, v_gt_, sigma_x_gt_, sigma_y_gt_, tau_xy_gt_ = elastic_body.ground_truth(x, y)

        bc.append(bc_)
        u_gt.append(u_gt_)
        v_gt.append(v_gt_)
        sigma_x_gt.append(sigma_x_gt_)
        sigma_y_gt.append(sigma_y_gt_)
        tau_xy_gt.append(tau_xy_gt_)

    # Convert to tensors
    bc = torch.tensor(bc, dtype=torch.float32, requires_grad=False)
    u_gt = torch.tensor(u_gt, dtype=torch.float32, requires_grad=False)
    v_gt = torch.tensor(v_gt, dtype=torch.float32, requires_grad=False)
    sigma_x_gt = torch.tensor(sigma_x_gt, dtype=torch.float32, requires_grad=False)
    sigma_y_gt = torch.tensor(sigma_y_gt, dtype=torch.float32, requires_grad=False)
    tau_xy_gt = torch.tensor(tau_xy_gt, dtype=torch.float32, requires_grad=False)

    x = torch.tensor(x, dtype=torch.float32).view(-1, 1)
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    x_copy = x.repeat(dataset_num, 1, 1).requires_grad_(False)
    y_copy = y.repeat(dataset_num, 1, 1).requires_grad_(False)

    # Initialize the network and optimizer
    pinn = PinnsFormer(d_model=64, d_hidden=64, n=2, heads=2, e=e, nu=nu).to(device)
    optimizer = LBFGS(pinn.parameters(), lr=1e-1, line_search_fn='strong_wolfe')
    loss_func = PinnLoss(e)

    # Training loop
    for epoch in range(epochs):
        # get the random indices of the batch
        indices = random_choice(dataset_num, batch_size)
        # get the batch data
        x_batch = x_copy[indices].to(device).requires_grad_(True)
        y_batch = y_copy[indices].to(device).requires_grad_(True)
        bc_batch = bc[indices].to(device)
        # get the ground truth of this batch
        u_gt_batch = u_gt[indices].unsqueeze(-1).to(device)
        v_gt_batch = v_gt[indices].unsqueeze(-1).to(device)
        sigma_x_gt_batch = sigma_x_gt[indices].unsqueeze(-1).to(device)
        sigma_y_gt_batch = sigma_y_gt[indices].unsqueeze(-1).to(device)
        tau_xy_gt_batch = tau_xy_gt[indices].unsqueeze(-1).to(device)

        def closure():
            # Forward pass
            u, v, epsilon_x, epsilon_y, gamma_xy, sigma_x, sigma_y, tau_xy = pinn(x_batch, y_batch, bc_batch)
            # Calculate the loss
            loss = loss_func(x_batch, y_batch, u, v, sigma_x, sigma_y, tau_xy, bc_batch, u_gt_batch, v_gt_batch,
                             sigma_x_gt_batch, sigma_y_gt_batch, tau_xy_gt_batch)
            print("Epoch: ", epoch, ", Loss: ", loss.cpu().detach().numpy())
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            return loss

        optimizer.step(closure)

    # Save the model
    torch.save(pinn, "./pinn.pth")


if __name__ == "__main__":
    main()
