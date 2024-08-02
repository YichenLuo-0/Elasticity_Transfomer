import torch


class PinnLoss:
    def __init__(self):
        return

    def __call__(self, x, y, sigma_x, sigma_y, tau_xy, bc):
        pde_loss = self.cauchy_equilibrium(x, y, sigma_x, sigma_y, tau_xy)
        bc_loss = self.boundary_condition(sigma_x, sigma_y, tau_xy, bc)
        return pde_loss + bc_loss

    def cauchy_equilibrium(self, x, y, sigma_x, sigma_y, tau_xy):
        grad_outputs = torch.ones_like(x)

        # 柯西平衡方程损失项
        dsigma_x_dx = torch.autograd.grad(sigma_x, x, grad_outputs=grad_outputs, create_graph=True)[0]
        dtau_xy_dy = torch.autograd.grad(tau_xy, y, grad_outputs=grad_outputs, create_graph=True)[0]
        eq1 = dsigma_x_dx + dtau_xy_dy

        dsigma_y_dy = torch.autograd.grad(sigma_y, y, grad_outputs=grad_outputs, create_graph=True)[0]
        dtau_xy_dx = torch.autograd.grad(tau_xy, x, grad_outputs=grad_outputs, create_graph=True)[0]
        eq2 = dtau_xy_dx + dsigma_y_dy
        return torch.mean(eq1 ** 2) + torch.mean(eq2 ** 2)

    def boundary_condition(self, sigma_x, sigma_y, tau_xy, bc):
        l = bc[:, :, 0:1]
        m = bc[:, :, 1:2]
        fx = bc[:, :, 2:3]
        fy = bc[:, :, 3:4]
        # 过滤出边界点及其边界条件
        mask = bc[:, :, 4:5] == 1

        sigma_x_bc = sigma_x[mask]
        sigma_y_bc = sigma_y[mask]
        tau_xy_bc = tau_xy[mask]
        l_bc = l[mask]
        m_bc = m[mask]
        fx_bc = fx[mask]
        fy_bc = fy[mask]

        # 计算边界条件损失项
        bc1 = (l_bc * sigma_x_bc) + (m_bc * tau_xy_bc) - fx_bc
        bc2 = (l_bc * tau_xy_bc) + (m_bc * sigma_y_bc) - fy_bc
        return torch.mean(bc1 ** 2) + torch.mean(bc2 ** 2)
