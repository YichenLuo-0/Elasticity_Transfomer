import torch


class PinnLoss:
    def __init__(self, E):
        self.E = E
        return

    def __call__(self, x, y, u, v, sigma_x, sigma_y, tau_xy, bc):
        pde_loss = self.cauchy_equilibrium(x, y, sigma_x, sigma_y, tau_xy)
        bc_loss = self.disp_boundary(u, v, bc) + self.force_boundary(sigma_x, sigma_y, tau_xy, bc)
        return pde_loss + bc_loss

    def cauchy_equilibrium(self, x, y, sigma_x, sigma_y, tau_xy):
        grad_outputs = torch.ones_like(x)

        # dσ_x/dx + dτ_xy/dy = 0
        dsigma_x_dx = torch.autograd.grad(sigma_x, x, grad_outputs=grad_outputs, create_graph=True)[0]
        dtau_xy_dy = torch.autograd.grad(tau_xy, y, grad_outputs=grad_outputs, create_graph=True)[0]
        eq1 = dsigma_x_dx + dtau_xy_dy

        # dτ_xy/dx + dσ_y/dy = 0
        dsigma_y_dy = torch.autograd.grad(sigma_y, y, grad_outputs=grad_outputs, create_graph=True)[0]
        dtau_xy_dx = torch.autograd.grad(tau_xy, x, grad_outputs=grad_outputs, create_graph=True)[0]
        eq2 = dtau_xy_dx + dsigma_y_dy
        return torch.sum(eq1 ** 2) + torch.sum(eq2 ** 2)

    def disp_boundary(self, u, v, bc):
        us = bc[:, :, 4:5]
        vs = bc[:, :, 5:6]

        # Filter out the displacement boundary points
        mask = bc[:, :, 6:7] == 1
        # Calculate loss only for the boundary points
        u_bc = u[mask]
        v_bc = v[mask]
        us_bc = us[mask]
        vs_bc = vs[mask]

        # Loss: (u_bc - us_bc) * E, (v_bc - vs_bc) * E
        # 由于位移边界的值非常小，我们需要将其和杨氏模量相乘，以增加其影响
        bc1 = (u_bc - us_bc) * self.E
        bc2 = (v_bc - vs_bc) * self.E
        return torch.sum(bc1 ** 2) + torch.sum(bc2 ** 2)

    def force_boundary(self, sigma_x, sigma_y, tau_xy, bc):
        l = bc[:, :, 0:1]
        m = bc[:, :, 1:2]
        fx = bc[:, :, 2:3]
        fy = bc[:, :, 3:4]

        # Filter out the force boundary points
        mask = bc[:, :, 6:7] == 2
        # Calculate loss only for the boundary points
        sigma_x_bc = sigma_x[mask]
        sigma_y_bc = sigma_y[mask]
        tau_xy_bc = tau_xy[mask]
        l_bc = l[mask]
        m_bc = m[mask]
        fx_bc = fx[mask]
        fy_bc = fy[mask]

        # Loss: (l * σ_x) + (m * τ_xy) = fx, (l * τ_xy) + (m * σ_y) = fy
        bc1 = (l_bc * sigma_x_bc) + (m_bc * tau_xy_bc) - fx_bc
        bc2 = (l_bc * tau_xy_bc) + (m_bc * sigma_y_bc) - fy_bc
        return torch.sum(bc1 ** 2) + torch.sum(bc2 ** 2)
